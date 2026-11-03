"""Connections to the external free services: Cohere and Pinecone."""
from __future__ import annotations

from llama_index.core import StorageContext, VectorStoreIndex
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


def document_embedding(settings: Settings) -> CohereEmbedding:
    return CohereEmbedding(
        api_key=_require(settings.cohere_api_key, "COHERE_API_KEY"),
        model_name=settings.embed_model,
        input_type="search_document",
    )


def query_embedding(settings: Settings) -> CohereEmbedding:
    # Cohere embeds questions differently from documents, so each side needs its own setting.
    return CohereEmbedding(
        api_key=_require(settings.cohere_api_key, "COHERE_API_KEY"),
        model_name=settings.embed_model,
        input_type="search_query",
    )


def get_llm(settings: Settings) -> CohereChatLLM:
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


def rebuild_vector_index(settings: Settings, chunks: list[TextNode]) -> int:
    """Replace the namespace's contents, so that deleted files do not leave stale vectors."""
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
    index = VectorStoreIndex.from_vector_store(
        get_vector_store(settings),
        embed_model=query_embedding(settings),
    )
    return index.as_retriever(similarity_top_k=settings.top_k)
