from .Indexer import Indexer
from .Retriever import Retriever
from pathlib import Path


class CLI:
    """ RAG CLI — each method become a command """

    def index(self, max_chunk_size: int = 2000,
              repo_path: str = "data/raw/vllm-0.10.1",
              out_dir: str = "data/processed") -> None:

        indexer = Indexer(dir_path=Path(repo_path))
        indexer.get_interest_paths()
        indexer.process_files(max_chunk_size)
        indexer.generate_chunks_and_tokenisation(out_dir)

    def search(self, query: str, k: int = 5) -> None:
        try:
            retriever = Retriever.load()
            retriever.retrieve_one_question(query, k)
        except FileNotFoundError:
            print("Index not found : use the command 'index' first")
            return

    def search_dataset(self,
                       dataset_path: str = "data/datasets/"
                       "UnansweredQuestions/dataset_docs_public.json",
                       k: int = 10,
                       save_directory: str =
                       "data/output/search_results") -> None:
        retriever = Retriever.load()

    def answer(self, question: str, k: int = 10) -> None:
        ...

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str =
                       "data/output/search_results_and_answer") -> None:
        ...

    def evaluate(self, student_answer_path: str, dataset_path: str,
                 k: int = 10, max_context_length: int = 2000) -> None:
        ...
