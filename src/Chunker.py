from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import List
from .Models import MinimalSource
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (MarkdownTextSplitter,
                                      RecursiveCharacterTextSplitter, Language)
from langchain_core.documents import Document
import re
import bm25s
import json


class Chunker(BaseModel):
    """Walk a source directory, load the ``.py``, ``.md`` and ``.txt`` files
    of interest, split them into chunks and persist them together with a BM25
    index that can later be reused by the retriever."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_path: Path
    chunks_list: list[MinimalSource] = []
    py_files: List[Document] = []
    md_files: List[Document] = []
    txt_files: list[Document] = []

    def load_docs(self):
        """Load every Markdown, Python and text file from the source directory
        into the corresponding ``md_files``/``py_files``/``txt_files`` lists."""
        md_loader = DirectoryLoader(
            "data/raw/vllm-0.10.1",
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
            )
        self.md_files = md_loader.load()

        py_loader = DirectoryLoader(
            "data/raw/vllm-0.10.1",
            glob="**/*.py",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        self.py_files = py_loader.load()

        txt_loader = DirectoryLoader(
            "data/raw/vllm-0.10.1",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        self.txt_files = txt_loader.load()

    def chunk_docs(self, max_chunk_size: int = 2000):
        """Split the loaded documents with the splitter matching their
        language and fill ``chunks_list`` with the resulting chunks converted
        into :class:`MinimalSource` instances (with original file offsets)."""

        md_splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=500
        )

        py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=500
        )
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=500
        )

        langchain_md_chunks: List[Document] = md_splitter.split_documents(
            self.md_files)

        langchain_py_chunks: List[Document] = py_splitter.split_documents(
            self.py_files)

        langchain_txt_chunks = txt_splitter.split_documents(self.txt_files)

        all_langchain_chunks: List[Document] = (langchain_md_chunks +
                                                langchain_py_chunks +
                                                langchain_txt_chunks)

        for chunk in all_langchain_chunks:
            file_path = chunk.metadata["source"]
            text = chunk.page_content

            with open(file_path, "r", encoding="utf-8") as f:
                full_file_content = f.read()

            start_idx = full_file_content.find(text)
            end_idx = start_idx + len(text) if start_idx != -1 else 0

            new_minimal_source = MinimalSource(
                text=text,
                file_path=file_path,
                first_character_index=start_idx,
                last_character_index=end_idx
                )
            self.chunks_list.append(new_minimal_source)

    def normalize(self, chunk_text: str):
        """Split camelCase identifiers and replace underscores with spaces so
        that the BM25 tokenizer can index sub-words as separate terms."""
        chunk_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", chunk_text)
        return chunk_text.replace("_", " ")

    def generate_chunks_and_tokenisation(self, out_dir: str):
        """Persist ``chunks_list`` as ``chunks.json`` and build/save the BM25
        index of the normalized chunk texts under ``out_dir``."""
        out = Path(out_dir)
        chunks_dir = out / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks = [chunk.model_dump() for chunk in self.chunks_list]
        (Path("data/processed/chunks") / "chunks.json").write_text(json.dumps(
            chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(chunks)} chunks stored in "
              "data/processed/chunks/chunks.json")

        corpus = [self.normalize(c.text) for c in self.chunks_list]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str(Path("data/processed") / "bm25_index"))
