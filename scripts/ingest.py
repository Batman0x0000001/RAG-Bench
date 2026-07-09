from __future__ import annotations

import argparse
import json
import logging

from src.ingestion.parse_documents import (
    extract_archives,
    parse_documents,
    summarize_documents,
    write_manifest,
)
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse EnterpriseRAG-Bench json/txt documents.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument(
        "--path",
        default=None,
        help="Document subdirectory or file path. Relative paths are resolved under data.documents_dir.",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help="Default source folder when --path is omitted. Defaults to data.default_source_type or github.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max source files to parse.")
    parser.add_argument("--extract", action="store_true", help="Extract zip archives before parsing json/txt files.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    data_config = config["data"]
    source_type = args.source_type or data_config.get("default_source_type", "github")

    if args.extract:
        extracted = extract_archives(data_config["archives_dir"], data_config["documents_dir"])
        logging.info("Extracted %s json/txt files from archives", extracted)

    documents = parse_documents(
        data_config["documents_dir"],
        limit=args.limit,
        path=args.path,
        source_type=source_type,
    )
    write_manifest(documents, data_config["manifest_file"])
    summary = summarize_documents(documents)
    logging.info(
        "Parsed %s chunks from %s source files into %s",
        summary["total_chunks"],
        summary["total_files"],
        data_config["manifest_file"],
    )
    logging.info("Ingestion summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
