from pydantic import BaseModel, ConfigDict
import bm25s
import json
from pathlib import Path
from .Models import (MinimalSource, MinimalSearchResults,
                     UnansweredQuestion, StudentSearchResults)
from typing import List
import re
import Stemmer


class Retriever(BaseModel):
    """BM25 retriever: load a saved index and return the most relevant chunks
    for a question"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    bm25: bm25s.BM25
    chunks: list[MinimalSource] = []

    @classmethod
    def load(cls, processed_dir: str = "data/processed") -> "Retriever":
        """Load the saved BM25 index and chunks file and return a ready-to-use
        Retriever"""
        bm25 = bm25s.BM25.load(str(Path(processed_dir) / "bm25_index"))

        with open(Path(processed_dir) / "chunks" / "chunks.json",
                  encoding="utf-8") as file:
            data = json.load(file)
        chunks = [MinimalSource(**dict) for dict in data]

        return cls(bm25=bm25, chunks=chunks)

    def get_best_sources(self, query: UnansweredQuestion,
                         k: int = 5) -> MinimalSearchResults:
        """Search the corpus for the query and return its top k chunks as a
        MinimalSearchResults"""
        normalized_question = self.normalize_query(query.question)
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(normalized_question, stopwords="en",
                                      stemmer=stemmer, show_progress=False)
        indexs, _ = self.bm25.retrieve(query_tokens, k=k)

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
        """Run retrieval on every question in the dataset and save the top k
        chunks per question as a StudentSearchResults JSON file under
        save_directory"""
        with open(dataset_path, encoding="utf-8") as file:
            data = json.load(file)
        questions_list: List[UnansweredQuestion] = []
        all_results: List[MinimalSearchResults] = []

        for q in data["rag_questions"]:
            new_question = UnansweredQuestion(question_id=q["question_id"],
                                              question=q["question"])
            questions_list.append(new_question)

        for question in questions_list:
            result_search: MinimalSearchResults = self.get_best_sources(
                question, k)
            all_results.append(result_search)

        path_output_dir = Path(save_directory)
        path_output_dir.mkdir(parents=True, exist_ok=True)
        file_name_dataset = Path(dataset_path).name
        search_results = StudentSearchResults(search_results=all_results, k=k)

        exclude_text = {
            "search_results": {
                "__all__": {"retrieved_sources": {"__all__": {"text"}}}
            }
        }
        with open(path_output_dir / file_name_dataset, 'w',
                  encoding="utf-8") as output_file:
            json.dump(search_results.model_dump(exclude=exclude_text),
                      output_file, indent=4, ensure_ascii=False)
        print("Saved student_search_results to "
              f"{path_output_dir}/{file_name_dataset}")

    def normalize_query(self, query_text: str) -> str:
        """Apply the same dual expansion as indexing: keep the original words
        and also add their camelCase/snake_case split"""
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", query_text)
        split = split.replace("_", " ")
        return query_text + " " + split
