*This project has been created as part of the 42 curriculum by evarache.*

# RAG against the machine

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system built on top
of the [vLLM](https://github.com/vllm-project/vllm) codebase. Given a natural
language question about vLLM, the system:

1. **Ingests** the repository and builds a searchable knowledge base.
2. **Retrieves** the most relevant code snippets and documentation chunks for a
   query.
3. **Answers** the question with a small local LLM
   (`Qwen/Qwen3-0.6B`) using only the retrieved context.
4. **Evaluates** the retrieval quality with a Recall@k metric.

The whole pipeline is exposed through a Python Fire CLI with one command per
stage (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`,
`evaluate`).

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) as package / project manager

### Installation

```bash
uv venv && uv sync
```

The raw vLLM repository must be available under `data/raw/vllm-0.10.1` (it is
provided as an attachment by the subject).

### Makefile

Common tasks are wired through the `Makefile`:

```bash
make install     # install dependencies
make run         # run the CLI entry point
make lint        # flake8 + mypy
make clean       # remove caches and temporary files
```

### Running the CLI

The CLI entry point is `python -m src`. All commands accept `--help`:

```bash
uv run python -m src --help
uv run python -m src <command> --help
```

## System architecture

The project is organised around five small components, each in its own module
under `src/`:

| Module | Class | Role |
|--------|-------|------|
| `Chunker.py` | `Chunker` | Loads `.py`, `.md` and `.txt` files, splits them into chunks and persists the BM25 index. |
| `Retriever.py` | `Retriever` | Reloads the BM25 index and returns the top-k most relevant chunks for a query. |
| `LLM.py` | `LLM` | Wraps the `Qwen/Qwen3-0.6B` text-generation pipeline to turn retrieved context into an answer. |
| `Evaluator.py` | `Evaluator` | Compares student predictions against ground truth and computes Recall@k using IoU on character offsets. |
| `Models.py` | pydantic models | Type-safe data containers shared by every stage. |
| `CommandLineInterface.py` | `CLI` | Glues the components together — each public method becomes a CLI sub-command via `python-fire`. |

### Pipeline flow

```
data/raw/vllm-0.10.1
        │
        ▼
   [ Chunker ] ──► data/processed/chunks/chunks.json
        │            data/processed/bm25_index/
        ▼
   [ Retriever ] ──► StudentSearchResults JSON
        │
        ▼
   [ LLM ] ──► StudentSearchResultsAndAnswer JSON
        │
        ▼
   [ Evaluator ] ──► Recall@k score
```

### Data models

All inputs and outputs are validated through the pydantic models defined in
`src/Models.py`:

- `MinimalSource` / `MinimalSourceOutput` — a retrieved chunk located by
  `file_path` + character offsets.
- `UnansweredQuestion` / `AnsweredQuestion` — questions, with or without their
  ground-truth answer and sources.
- `MinimalSearchResults` / `MinimalAnswer` — retrieval output, optionally
  extended with a generated answer.
- `StudentSearchResults` / `StudentSearchResultsAndAnswer` — top-level JSON
  artefacts produced by `search_dataset` and `answer_dataset`.
- `RagDataset` — a list of questions used as input or ground truth.

## Chunking strategy

The chunker uses **language-aware splitters** from
`langchain_text_splitters`, with a configurable maximum chunk size (default
`2000` characters) and a 500-character overlap (25%):

- **Markdown** (`.md`) → `MarkdownTextSplitter`: respects headers so a chunk
  rarely spans two unrelated sections.
- **Python** (`.py`) → `RecursiveCharacterTextSplitter.from_language(PYTHON)`:
  prefers splitting on class / function boundaries, then statements, then
  lines.
- **Text** (`.txt`) → generic `RecursiveCharacterTextSplitter`.

For every produced chunk we reopen the original file and recover the exact
`first_character_index` / `last_character_index` so that the output is
directly comparable to the ground-truth character spans used by the
evaluator.

Each chunk is then turned into a **searchable representation** before being
fed to the BM25 tokenizer (detailed in *Retrieval method*): the body text is
kept both as-is and split on camelCase / underscores, and the chunk's
file-path tokens (e.g. `lora`, `fused_batched_moe`) are appended and weighted,
because the file name is a strong topic signal.

## Retrieval method

Retrieval is done with **BM25** through the
[`bm25s`](https://github.com/xhluca/bm25s) library:

1. At indexing time, each chunk's searchable text (body kept both raw and
   split on camelCase / underscores, plus its boosted file-path tokens) is
   tokenized with English stopword removal and **Snowball stemming**
   (PyStemmer); the resulting BM25 index is persisted under
   `data/processed/bm25_index/`.
2. At query time, `Retriever.get_best_sources` applies the **exact same**
   expansion, stopword removal and stemming to the question, calls
   `bm25.retrieve` and returns the top-k chunks wrapped in a
   `MinimalSearchResults`.
3. For datasets, `get_best_sources_dataset` iterates over every
   `UnansweredQuestion` and writes a single `StudentSearchResults` JSON file.

This identifier-aware expansion (dual raw/split tokens + stemming + file-path
boosting) is what lifts recall@5 to **86% (docs)** and **80% (code)**, well
above the 80% / 50% thresholds.

BM25 was chosen over TF-IDF for its better behaviour on long documents and
over dense embeddings for its zero-GPU footprint and very fast cold start —
both criteria explicitly required by the subject (5 min indexing budget,
60 s cold start).

## Answer generation

`LLM.generate_answer` builds a Qwen-style chat prompt with the top retrieved
source as context and a strict system message asking the model to answer
*only* from the context, or to say "I don't know". The generated text is then
packaged together with the source pointers into a
`StudentSearchResultsAndAnswer` object.

`LLM.handle_dataset` does the same in batch over a previously produced
`StudentSearchResults` file, with a `tqdm` progress bar.

## Performance analysis

Recall@k is computed by `Evaluator.evaluate`:

- For each question, the top-k retrieved sources are compared to the
  ground-truth sources.
- A predicted source counts as a hit when it shares the same `file_path` as a
  ground-truth source **and** has an IoU (Intersection over Union of
  character ranges) of at least **5%** with it.
- The reported metric is the percentage of questions for which at least one
  hit was found.

Target thresholds from the subject:

| Dataset | Metric   | Threshold | Measured |
|---------|----------|-----------|----------|
| Docs    | Recall@5 | >= 80%    | **86%**  |
| Code    | Recall@5 | >= 50%    | **80%**  |

The official scores must be measured with the provided moulinette (see
*Example usage* below).

## Design decisions

- **BM25 over dense retrieval**: smaller, faster, no GPU, and good enough for
  identifier-heavy corpora like vLLM.
- **Language-aware splitters**: keeping Python classes/functions together
  drastically improves recall on code questions.
- **Identifier-aware retrieval**: each chunk and query is expanded (original
  text + camelCase/underscore split), stemmed with Snowball, and the chunk's
  file-path tokens are appended and boosted. This makes BM25 robust to the
  wording gap between natural-language questions and source code, and is what
  pushes recall past the thresholds (86% docs / 80% code).
- **One pydantic model per artefact**: each JSON file produced by the
  pipeline is validated at read and write time, which removes a whole class
  of "silent schema drift" bugs between commands.
- **`python-fire` CLI**: each method of the `CLI` class becomes a command
  automatically — adding a new stage is a single method to write.

## Challenges faced

- **Recovering character offsets after splitting**: `langchain` splitters
  return text only, so we re-locate every chunk in its original file with
  `str.find` and store the resulting indices. This is what makes the IoU
  evaluation possible.
- **Matching identifier queries**: questions often use natural English while
  the corpus is full of `snake_case` / `camelCase` names. Normalising both
  sides before BM25 indexing was key to reaching the recall threshold on
  code.
- **Cold start with `transformers`**: loading `Qwen/Qwen3-0.6B` is expensive,
  so the model is **loaded lazily** (only the `answer` / `answer_dataset`
  commands build the pipeline, on first use) — `index`, `search` and
  `evaluate` never touch it, keeping their cold start well under the 60 s
  budget. `transformers` logging is silenced so generation stays quiet.

## Example usage

### Index the vLLM repository

```bash
uv run python -m src index --max_chunk_size 2000
```

### Search a single query

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

### Search all questions of a dataset

Documentation questions (by default):

```bash
uv run python -m src search_dataset
```

Code questions:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json
```

### Answer a single query

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

### Generate answers for a whole dataset

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer
```

### Evaluate retrieval with evaluate command
```bash
uv run python -m src evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Evaluate retrieval with the moulinette

```bash
cd moulinette
./moulinette-ubuntu evaluate_student_search_results \
    --student_answer_path ../data/output/search_results/dataset_docs_public.json \
    --dataset_path ../data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## Resources

### Documentation and articles

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [BM25 — original Okapi BM25 paper](https://en.wikipedia.org/wiki/Okapi_BM25)
- [`bm25s` — fast pure-Python BM25 implementation](https://github.com/xhluca/bm25s)
- [LangChain text splitters documentation](https://python.langchain.com/docs/concepts/text_splitters/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Fire — automatic CLI generation](https://github.com/google/python-fire)

### Use of AI

AI assistants were used during this project for the following, well-scoped
tasks:

- **Exploring the LangChain text-splitter API** and choosing the right
  splitter per file type (Markdown vs. Python vs. plain text).
- **Drafting docstrings** for the classes and methods of the `src/` modules
  (every docstring was then re-read and edited by hand).
- **Brainstorming the chunk-normalisation step** (camelCase / snake_case
  handling) before benchmarking it against the moulinette.
- **Rubber-ducking design decisions** (BM25 vs. dense retrieval, prompt shape
  for the Qwen chat template).

No AI-generated code was committed without being read, understood and
adapted to the project's structure.
