"""Check the API keys with one small call to each service: python -m agentic_rag.check_services"""
from __future__ import annotations

from pinecone import Pinecone

from .config import enable_netfree_ssl_if_available, load_settings
from .services import get_llm, query_embedding, uses_local_store


def main() -> int:
    enable_netfree_ssl_if_available()
    settings = load_settings()
    provider = settings.llm_provider.capitalize()
    checks = {
        f"{provider} embeddings ({settings.embed_model})": lambda: len(
            query_embedding(settings).get_query_embedding("בדיקה")
        ),
        f"{provider} chat ({settings.llm_model})": lambda: get_llm(settings)
        .complete("Reply with: OK")
        .text.strip(),
    }
    if uses_local_store(settings):
        print(f"INFO  No PINECONE_API_KEY: vectors are kept locally in {settings.local_vector_dir}")
    else:
        checks["Pinecone"] = lambda: (
            f"{len(Pinecone(api_key=settings.pinecone_api_key).list_indexes())} indexes"
        )
    failed = False
    for name, check in checks.items():
        try:
            print(f"OK    {name}: {check()}")
        except Exception as error:
            failed = True
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
