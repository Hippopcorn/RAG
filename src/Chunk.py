from pydantic import BaseModel
from pathlib import Path


class Chunk(BaseModel):
    id: int
    text: str
    file_path: str
    first_index: int
    last_index: int


class Indexer(BaseModel):
    dir_path: Path = Path("vllm-0.10.1")

    def get_interest_paths(self):
        """ Get a filtered list with the paths of the .py and .md files
            located in the dir_path """
        filtered_paths: list[Path] = []

        for file_path in self.dir_path.rglob("*.py"):

            if "tests" in file_path.parts or "__pycache__" in file_path.parts:
                continue

            filtered_paths.append(file_path)

        for file_path in self.dir_path.rglob("*.md"):

            if "tests" in file_path.parts or "__pycache__" in file_path.parts:
                continue

            filtered_paths.append(file_path)

        return filtered_paths
