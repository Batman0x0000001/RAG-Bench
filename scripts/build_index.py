from __future__ import annotations

import argparse
import logging

from src.indexing.qdrant_index import index_documents, recreate_collection
from src.ingestion.parse_documents import read_manifest
from src.retrieval.embeddings import build_embeddings
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant vector index.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    documents = read_manifest(config["data"]["manifest_file"], limit=args.limit)

    embeddings = build_embeddings(config["embedding"])
    if not args.no_recreate:
        recreate_collection(config["qdrant"])
    index_documents(documents, embeddings, config["qdrant"], batch_size=args.batch_size)
    logging.info("Indexed %s documents into Qdrant collection %s", len(documents), config["qdrant"]["collection"])


if __name__ == "__main__":
    main()
