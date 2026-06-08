from pydantic import BaseModel, ConfigDict
import bm25s
import json
from pathlib import Path
from .Indexer import Chunk
from .Models import MinimalSource, MinimalSearchResults, UnansweredQuestion
from typing import List


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

    def retrieve_one_question(self, query: UnansweredQuestion, k: int = 5):
        """ """
        indexs, scores = self.bm25.retrieve(bm25s.tokenize(
            query.question), k=k)

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

        result = MinimalSearchResults(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=retrieved_sources
        )
        print(result.model_dump_json(indent=2))

    def retrieve_dataset(self, dataset_path: str, k: int = 5):
        with open(dataset_path, encoding="utf-8") as file:
            data = json.load(file)
        questions_list: List[UnansweredQuestion] = []

        for q in data["rag_questions"]:
            new_question = UnansweredQuestion(question_id=q["question_id"],
                                              question=q["question"])
            questions_list.append(new_question)
        
        for question in questions_list:
            self.retrieve_one_question(question, k)
