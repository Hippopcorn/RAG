from pydantic import BaseModel, ConfigDict
import bm25s
import json
from .Indexer import Chunk
from pathlib import Path


class Retriever(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    bm25: bm25s.BM25
    chunks: list[Chunk] = []

    @classmethod
    def load(cls, processed_dir: str = "data/processed") -> "Retriever":
        """Reload the BM25 index and chunks saved by `index`."""
        bm25 = bm25s.BM25.load(str(Path(processed_dir) / "bm25_index"))

        with open(Path(processed_dir) / "chunks" / "chunks.json",
                  encoding="utf-8") as file:
            data = json.load(file)
        chunks = [Chunk(**dico) for dico in data]

        return cls(bm25=bm25, chunks=chunks)
