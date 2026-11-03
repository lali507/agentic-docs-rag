"""Connections to the external services: OpenAI or Cohere, and Pinecone (or a local store)."""
from __future__ import annotations

import shutil

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .cohere_llm import CohereChatLLM
from .config import Settings


class MissingKeyError(RuntimeError):
    """A required API key is not configured."""


def _require(value: str, name: str) -> str:
    if not value:
        raise MissingKeyError(f"{name} is missing. Copy .env.example to .env and fill it in.")
    return value


def _openai_embedding(settings: Settings) -> BaseEmbedding:
    from llama_index.embeddings.openai import OpenAIEmbedding

    # OpenAI embeds questions and documents the same way.
    return OpenAIEmbedding(
        api_key=_require(settings.openai_api_key, "OPENAI_API_KEY"),
        model=settings.embed_model,
        dimensions=settings.embed_dimension,
    )


def document_embedding(settings: Settings) -> BaseEmbedding:
    if settings.llm_provider == "openai":
        return _openai_embedding(settings)
    return CohereEmbedding(
        api_key=_require(settings.cohere_api_key, "COHERE_API_KEY"),
        model_name=settings.embed_model,
        input_type="search_document",
    )


def query_embedding(settings: Settings) -> BaseEmbedding:
    if settings.llm_provider == "openai":
        return _openai_embedding(settings)
    # Cohere embeds questions differently from documents, so each side needs its own setting.
    return CohereEmbedding(
        api_key=_require(settings.cohere_api_key, "COHERE_API_KEY"),
        model_name=settings.embed_model,
        input_type="search_query",
    )


def get_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            api_key=_require(settings.openai_api_key, "OPENAI_API_KEY"),
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=4000,
            max_retries=3,
        )
    return CohereChatLLM(
        api_key=_require(settings.cohere_api_key, "COHERE_API_KEY"),
        model=settings.llm_model,
        temperature=0.0,
    )


def _pinecone_index(settings: Settings, create: bool):
    client = Pinecone(api_key=_require(settings.pinecone_api_key, "PINECONE_API_KEY"))
    if create and not client.has_index(settings.pinecone_index):
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.embed_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return client.Index(settings.pinecone_index)


def get_vector_store(settings: Settings, create: bool = False) -> PineconeVectorStore:
    return PineconeVectorStore(
        pinecone_index=_pinecone_index(settings, create),
        namespace=settings.pinecone_namespace,
    )


def uses_local_store(settings: Settings) -> bool:
    return not settings.pinecone_api_key


def _rebuild_local_index(settings: Settings, chunks: list[TextNode]) -> int:
    directory = settings.local_vector_dir
    if directory.exists():
        shutil.rmtree(directory)
    index = VectorStoreIndex(chunks, embed_model=document_embedding(settings), show_progress=True)
    index.storage_context.persist(persist_dir=str(directory))
    return len(chunks)


def _local_retriever(settings: Settings):
    directory = settings.local_vector_dir
    if not (directory / "docstore.json").exists():
        raise MissingKeyError(
            "The local vector index was not built yet. Run: python -m agentic_rag.prepare"
        )
    storage = StorageContext.from_defaults(persist_dir=str(directory))
    index = load_index_from_storage(storage, embed_model=query_embedding(settings))
    return index.as_retriever(similarity_top_k=settings.top_k)


def rebuild_vector_index(settings: Settings, chunks: list[TextNode]) -> int:
    """Replace the namespace's contents, so that deleted files do not leave stale vectors."""
    if uses_local_store(settings):
        return _rebuild_local_index(settings, chunks)
    vector_store = get_vector_store(settings, create=True)
    try:
        vector_store.clear()
    except Exception as error:
        # A namespace that does not exist yet has nothing to clear.
        if getattr(error, "status", None) != 404 and "not found" not in str(error).lower():
            raise
    VectorStoreIndex(
        chunks,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=document_embedding(settings),
        show_progress=True,
    )
    return len(chunks)


def get_retriever(settings: Settings):
    if uses_local_store(settings):
        return _local_retriever(settings)
    index = VectorStoreIndex.from_vector_store(
        get_vector_store(settings),
        embed_model=query_embedding(settings),
    )
    return index.as_retriever(similarity_top_k=settings.top_k)
