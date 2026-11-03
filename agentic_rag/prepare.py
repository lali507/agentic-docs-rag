"""Build the vector index and the structured data: python -m agentic_rag.prepare"""
from __future__ import annotations

import argparse
import asyncio
import json

from .config import enable_netfree_ssl_if_available, load_settings
from .ingest_workflow import IngestWorkflow
from .services import MissingKeyError, get_llm, rebuild_vector_index


async def _run(skip_extraction: bool) -> int:
    settings = load_settings()
    llm = None if skip_extraction else get_llm(settings)
    workflow = IngestWorkflow(
        settings=settings,
        llm=llm,
        write_vectors=lambda chunks: rebuild_vector_index(settings, chunks),
    )
    result = await workflow.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-extraction", action="store_true", help="Build only the vector index (Stage A)"
    )
    args = parser.parse_args()
    enable_netfree_ssl_if_available()
    try:
        return asyncio.run(_run(args.skip_extraction))
    except MissingKeyError as error:
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
