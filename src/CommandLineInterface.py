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
    """ RAG CLI — each method become a command """

    def index(self, max_chunk_size: int = 2000,
              repo_path: str = "data/raw/vllm-0.10.1",
              out_dir: str = "data/processed") -> None:
        chunker = Chunker(dir_path=Path(repo_path))
        chunker.load_docs()
        chunker.chunk_docs(max_chunk_size)
        chunker.generate_chunks_and_tokenisation(out_dir)

    def search(self, query: str, k: int = 5) -> None:
        retriever = Retriever.load()
        query_catched = UnansweredQuestion(question_id="single_query",
                                           question=query)
        result = retriever.get_best_sources(query_catched, k)
        print(result.model_dump_json(indent=2))

    def search_dataset(self,
                       dataset_path: str = "data/datasets/"
                       "UnansweredQuestions/dataset_docs_public.json",
                       k: int = 5,
                       save_directory: str =
                       "data/output/search_results") -> None:
        retriever = Retriever.load()
        retriever.get_best_sources_dataset(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5) -> None:
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        llm = LLM()
        llm.handle_dataset(student_search_results_path, save_directory)

    def evaluate(self, student_answer_path: str = "data/output/"
                 "search_results/dataset_docs_public.json",
                 dataset_path: str = "data/datasets/AnsweredQuestions/"
                 "dataset_docs_public.json",
                 k: int = 10, max_context_length: int = 2000) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(student_answer_path, dataset_path, k)
