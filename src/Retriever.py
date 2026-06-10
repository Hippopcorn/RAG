from pydantic import BaseModel, ConfigDict
import bm25s
import json
from pathlib import Path
from .Models import (MinimalSource, MinimalSearchResults,
                     UnansweredQuestion, StudentSearchResults)
from typing import List, Dict
import re


class Retriever(BaseModel):
    """BM25-based retriever that loads a pre-built index and returns the most
    relevant chunks for a given question."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    bm25: bm25s.BM25
    chunks: list[MinimalSource] = []

    @classmethod
    def load(cls, processed_dir: str = "data/processed") -> "Retriever":
        """Reload the BM25 index and the chunks file produced by ``index`` and
        return a ready-to-use :class:`Retriever` instance."""
        bm25 = bm25s.BM25.load(str(Path(processed_dir) / "bm25_index"))

        with open(Path(processed_dir) / "chunks" / "chunks.json",
                  encoding="utf-8") as file:
            data = json.load(file)
        chunks = [MinimalSource(**dict) for dict in data]

        return cls(bm25=bm25, chunks=chunks)

    def get_best_sources(self, query: UnansweredQuestion,
                         k: int = 5) -> MinimalSearchResults:
        """Run BM25 retrieval for ``query`` and return a
        :class:`MinimalSearchResults` containing the ``k`` most relevant
        chunks of the corpus."""
        normalized_question = self.normalize_query(query.question)
        indexs, score = self.bm25.retrieve(bm25s.tokenize(
            normalized_question), k=k)

        best_chunk_indexs = indexs[0]

        retrieved_sources = []

        for i in best_chunk_indexs:
            retrieved_sources.append(self.chunks[i])

        result = MinimalSearchResults(
            question_id=query.question_id,
            question_str=query.question,
            retrieved_sources=retrieved_sources
        )
        return result

    def get_best_sources_dataset(self, dataset_path: str,
                                 save_directory: str, k: int = 5) -> None:
        """Load every question from the dataset at ``dataset_path``, run
        retrieval for each of them and serialize the aggregated results as a
        :class:`StudentSearchResults` JSON file under ``save_directory``."""
        with open(dataset_path, encoding="utf-8") as file:
            data = json.load(file)
        questions_list: List[UnansweredQuestion] = []
        all_results: List[MinimalSearchResults] = []
        dict_final: Dict = {"k": int(k)}

        for q in data["rag_questions"]:
            new_question = UnansweredQuestion(question_id=q["question_id"],
                                              question=q["question"])
            questions_list.append(new_question)

        for question in questions_list:
            result_search: MinimalSearchResults = self.get_best_sources(
                question, k)
            all_results.append(result_search)

        results_pure_python = [res.model_dump() for res in all_results]
        dict_final = {"search_results": results_pure_python} | dict_final

        path_output_dir = Path(save_directory)
        path_output_dir.mkdir(parents=True, exist_ok=True)
        file_name_dataset = Path(dataset_path).name
        search_results = StudentSearchResults(search_results=all_results, k=k)

        with open(path_output_dir / file_name_dataset, 'w',
                  encoding="utf-8") as output_file:
            json.dump(search_results.model_dump(), output_file, indent=4,
                      ensure_ascii=False)
        print("Saved student_search_results to "
              f"{path_output_dir}/{file_name_dataset}")

    def normalize_query(self, query_text: str) -> str:
        """Applique exactement le même traitement qu'à l'indexation."""
        query_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", query_text)
        return query_text.replace("_", " ")
