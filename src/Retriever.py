from pydantic import BaseModel, ConfigDict
import bm25s
import json
from pathlib import Path
from .Indexer import Chunk
from .Models import MinimalSource, SearchResults


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

    def retrieve_one_question(self, query: str, k: int = 5):
        """ """
        indexs, scores = self.bm25.retrieve(bm25s.tokenize(query), k=k)

        best_chunk_indexs = indexs[0]

        retrieved_sources = []
        for i in best_chunk_indexs:
            chunk = self.chunks[i]
            source = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_index,
                last_character_index=chunk.last_index
            )
            retrieved_sources.append(source)

        result = SearchResults(
            question_id="single_query",
            question=query,
            retrieved_sources=retrieved_sources
        )
        print(result.model_dump_json(indent=2))

    def retrieve_dataset(self, dataset_path: str):
        pass
