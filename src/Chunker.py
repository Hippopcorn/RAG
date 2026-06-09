from pydantic import BaseModel, ConfigDict
from pathlib import Path
from .Models import MinimalSource
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
# from tqdm import tqdm


class Indexer(BaseModel):
    """ Handles retrieving all the interest files from the vllm directory
        and index them into MinimalSource """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_path: Path
    chunks_list: list[MinimalSource] = []
    py_files: list[Document] = []
    md_files: list[Document] = []

    def load_docs(self):
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
