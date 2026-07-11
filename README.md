# EnterpriseRAG-Bench LangChain/LangGraph RAG

This project is a resume-friendly RAG benchmark scaffold for
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench).
It uses Qdrant for vector search, SiliconFlow for OpenAI-compatible embeddings,
DeepSeek's Anthropic-compatible API for reasoning and answer generation,
LangChain for standard retrieval/model/prompt components, and LangGraph for
adaptive workflow orchestration.

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
`BaseLoader -> Document -> RecursiveCharacterTextSplitter -> QdrantVectorStore`.
At query time, both Qdrant Dense search and the local BM25 index implement the
standard `BaseRetriever` interface.
Manifest rows use the standard document shape with `page_content` and `metadata`.

Benchmark execution always uses one unified adaptive graph. LangChain components
perform query planning, parallel `BaseRetriever.batch()` calls, prompt/model
execution, and output parsing. Each planned query runs against Dense and BM25
channels; chunk-level RRF combines those channels, then document-level RRF
combines all query variants. LangGraph controls the state transitions:
`plan -> hybrid retrieve -> query RRF -> rerank -> parent expand -> evidence check`.
When evidence is incomplete, the graph performs one bounded follow-up retrieval
round before answer generation. Follow-up queries use the same hybrid retriever.
Document budgets adapt from small single-source
queries to broad completeness queries. This does not require a different manifest
or vector collection.

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

The benchmark output is written to `runs/<RUN_NAME>/answers.jsonl`.
Adaptive planning, executed queries, retrieval rounds, selected documents, and
evidence checks are recorded in `runs/<RUN_NAME>/graph_traces.jsonl`. Each query
candidate also records whether it came from `dense`, `bm25`, or both channels.
The reranker receives document RRF rank, query-hit count, retrieval channels, and
evidence-first excerpts. For single-document questions that explicitly request a
wait-time value, HTTP status, size limit, or three named modes/settings, a narrow
deterministic guard prevents a similar-looking document without that requested signal
from replacing a candidate that contains direct evidence. Incidental phrases such as
`time limit` do not trigger the wait-time rule. Named-enumeration evidence must also
match the question topic near the enumeration, so unrelated component modes are ignored.
`rerank_history` records the model's original choice,
the final choice, its reason, and any guardrail decision for every retrieval round.
The answer node receives the latest safe retrieval assessment, checks every comparison
side and supported label, and permits only minimal inferences that follow directly from
documented semantics. An overridden reranker reason is never passed to answer generation.

The BM25 index is rebuilt in memory from the manifest when a benchmark process
starts. It adds startup time but creates no persistent files and does not modify
the existing Qdrant collection. To run a low-cost hybrid comparison first:

```powershell
$env:RUN_NAME = "12_generation_and_enum_fix_n10"
python -m scripts.run_benchmark --question-source-type github --limit 10
python -m scripts.evaluate --source-type github --parallelism 1 --only-answered
```

## Evaluation

The local evaluation wrapper keeps the official benchmark repository unchanged.
It filters the question set to GitHub, runs the official metrics evaluator, and
then writes retrieval diagnostics and an incorrect-answer worklist.

Configure the official judge first with `LLM_PROVIDER`, `LLM_API_KEY`,
`LLM_MODEL_NAME`, and `CHEAP_LLM_MODEL_NAME`, then run from this project root:

```bash
python -m scripts.run_benchmark --question-source-type github
python -m scripts.evaluate --source-type github --parallelism 3
```

On PowerShell, the evaluation variables can be loaded from the project `.env`
into the current terminal before evaluation:

```powershell
. .\scripts\activate_eval_env.ps1
python -m scripts.evaluate --source-type github --parallelism 1
```

The leading dot and space are required so the environment variables remain in
the current PowerShell session.

The run directory will contain `github_questions.jsonl`,
`official_results.json`, `supplementary_metrics.json`, and
`failed_questions.jsonl`. The default official run uses the original gold set.
Use `--official-correction` only when the three-judge document correction flow
is intentionally required, and `--resume` to continue an interrupted run.
For a partial `answers.jsonl`, add `--only-answered` to evaluate exactly the
question IDs present in that file.

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

- SiliconFlow embedding model defaults to `Qwen/Qwen3-Embedding-0.6B`, with
  `EMBEDDING_VECTOR_SIZE=1024`. If you change the embedding model, update the
  vector size and recreate the Qdrant collection.
