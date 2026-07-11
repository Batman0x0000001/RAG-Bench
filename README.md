# EnterpriseRAG-Bench LangChain/LangGraph RAG

This project is a resume-friendly RAG benchmark scaffold for
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench).
It uses Qdrant for vector search, SiliconFlow for OpenAI-compatible embeddings,
DeepSeek's Anthropic-compatible API for answer generation, LangChain for the
minimal RAG chain, and LangGraph for workflow orchestration.

## Setup

```bash
conda env create -f environment.yml
conda activate enterprise-rag-bench
copy .env.example .env
```

Fill these values in `.env`:

```text
DEEPSEEK_API_KEY=
SILICONFLOW_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise_rag_bench
```

Configuration now comes from code defaults plus `.env`. 

## Data Layout

Place the official EnterpriseRAG-Bench raw JSON files under
`data/raw/generated_data/sources`. Put the GitHub
source JSON files under `data/raw/generated_data/sources/github`.

Place `questions.jsonl` under `data/raw/questions.jsonl`.

## Quickstart

```bash
python -m scripts.ingest
python -m scripts.build_index
python -m scripts.run_benchmark
```

Ingestion writes a chunk-level manifest to
`data/processed/github_documents.jsonl`: one source JSON file can produce
multiple retrievable chunks, all sharing the same `dsid`.
The ingestion pipeline follows LangChain's modular retrieval interfaces:
`BaseLoader -> Document -> RecursiveCharacterTextSplitter -> QdrantVectorStore -> BaseRetriever`.
Manifest rows use the standard document shape with `page_content` and `metadata`.

For GitHub PR JSON files, chunks are grouped by PR semantics instead of raw
field names: `overview`, `description`, `discussion`, `release`, `changes`,
`ci`, and `post_merge`. Short structured fields are folded into the overview or
the matching section, while long fields such as descriptions and discussions are
split with section-specific rules.

By default, ingestion reads the `github` source folder and benchmark execution
only uses questions whose `source_types` include `github`. You can override the
document path explicitly:

```bash
python -m scripts.ingest --path github --limit 100
python -m scripts.ingest --path github/some_file.json
python -m scripts.run_benchmark --question-source-type github --limit 5
```

Use LangGraph mode when you want to exercise the graph workflow:

```bash
python -m scripts.run_benchmark --mode graph --limit 5
```

The benchmark output is written to `runs/<RUN_NAME>/answers.jsonl`.

## Index migration

The standard Document manifest format is not compatible with manifests created
by earlier versions of this project. Regenerate the manifest and rebuild the
collection together:

```bash
python -m scripts.ingest
python -m scripts.build_index
```

Rebuilding is required because the SiliconFlow/Qwen embedding path now sends
raw text to the provider tokenizer and indexed points use stable IDs derived
from each chunk's `chunk_id`.

## Notes

- The first version intentionally keeps the graph simple:
  `Start -> Retrieve -> Generate Answer -> Final`.
- Core modules include Chinese comments for learning-oriented readability.
- SiliconFlow embedding model defaults to `Qwen/Qwen3-Embedding-0.6B`, with
  `EMBEDDING_VECTOR_SIZE=1024`. If you change the embedding model, update the
  vector size and recreate the Qdrant collection.
