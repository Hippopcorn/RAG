from .Retriever import Retriever
from .Models import UnansweredQuestion, MinimalSearchResults
from .Evaluator import Evaluator
from .Chunker import Chunker
from pathlib import Path
from .LLM import LLM
import json
import logging
import warnings
import os


class CLI:
    """RAG command-line interface; each public method is a sub-command"""

    def index(self, max_chunk_size: int = 2000,
              repo_path: str = "data/raw/vllm-0.10.1",
              out_dir: str = "data/processed") -> None:
        """Load files from repo_path, split them into chunks and save the
        chunks file and the BM25 index under out_dir"""
        if not Path(repo_path).exists():
            print(f"Error: repo not found: {repo_path}")
            return
        chunker = Chunker(dir_path=Path(repo_path))
        chunker.load_docs()
        chunker.chunk_docs(max_chunk_size)
        chunker.generate_chunks_and_tokenisation(out_dir)

    def search(self, query: str, k: int = 5) -> None:
        """Run one query against the index and print the top k sources as
        JSON"""
        if not Path("data/processed/bm25_index").exists():
            print("Error: no index found, run 'index' first.")
            return
        if k <= 0:
            print("Error: k must be > 0")
            return
        retriever = Retriever.load()
        query_catched = UnansweredQuestion(question_id="single_query",
                                           question=query)
        result = retriever.get_best_sources(query_catched, k)
        print(result.model_dump_json(
            indent=2,
            exclude={"retrieved_sources": {"__all__": {"text"}}}))

    def search_dataset(self,
                       dataset_path: str = "data/datasets/"
                       "UnansweredQuestions/dataset_docs_public.json",
                       k: int = 5,
                       save_directory: str =
                       "data/output/search_results") -> None:
        """Run retrieval on every question in a dataset and save the top k
        results per question in save_directory."""
        if not Path("data/processed/bm25_index").exists():
            print("Error: no index found, run 'index' first.")
            return
        if k <= 0:
            print("Error: k must be > 0")
            return
        retriever = Retriever.load()
        retriever.get_best_sources_dataset(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve the top k sources for the query, generate an answer with
        the LLM and print it as JSON"""
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not Path("data/processed/bm25_index").exists():
            print("Error: no index found, run 'index' first.")
            return
        if k <= 0:
            print("Error: k must be > 0")
            return
        llm = LLM()
        retriever = Retriever.load()
        query_catched = UnansweredQuestion(question_id="q1",
                                           question=query)
        search_results: MinimalSearchResults = retriever.get_best_sources(
            query_catched, k)

        printable_answer = llm.generate_answer(
            search_results, query_catched).model_dump()
        print(json.dumps(printable_answer, indent=4, ensure_ascii=False))

    def answer_dataset(self, student_search_results_path: str =
                       "data/output/search_results/dataset_docs_public.json",
                       save_directory: str =
                       "data/output/search_results_and_answer") -> None:
        """Generate an LLM answer for every search result in the file and save
        the answered dataset under save_directory"""
        llm = LLM()
        llm.handle_dataset(student_search_results_path, save_directory)

    def evaluate(self, student_answer_path: str = "data/output/"
                 "search_results/dataset_docs_public.json",
                 dataset_path: str = "data/datasets/AnsweredQuestions/"
                 "dataset_docs_public.json",
                 k: int = 10) -> None:
        """Compare student results to the ground truth and print the
        retrieval Recall@k"""
        evaluator = Evaluator()
        evaluator.evaluate(student_answer_path, dataset_path, k)
