from pydantic import BaseModel
from pathlib import Path


class Chunk(BaseModel):
    id: int
    text: str
    file_path: str
    first_index: int
    last_index: int


class Indexer(BaseModel):
    """ Handles retrieving all the interest files from the vllm directotory
        and index them into Chunks """
    dir_path: Path = Path("vllm-0.10.1")
    list_chunks: list[Chunk] = []

    def get_interest_paths(self):
        """ Get a filtered list with the paths of the .py and .md files
            located in the dir_path """
        filtered_paths: list[Path] = []

        for file_path in self.dir_path.rglob("*.py"):

            if "tests" in file_path.parts or "__pycache__" in file_path.parts:
                continue

            filtered_paths.append(file_path)

        for file_path in self.dir_path.rglob("*.md"):

            if ("tests" in file_path.parts or "__pycache__" in file_path.parts
                    or "pytest_cache" in file_path.parts):
                continue

            filtered_paths.append(file_path)

        return filtered_paths

    def index_md_file(self, path: str):
        """ Read a md file, then split it at each \n\n, check alls splits and
            if there are two tittles following, merge them. Call the
            split_oversized_block function to recut splitted_blocs bigger than
            2000 characters. Then, create blocks with titles and text, until
            2000 characters """
        chunks: list[Chunk] = []
        actual_block: str = ""
        blocks_list: list[str] = []
        fusion_title_list: list[str] = []

        try:
            with open(path, "r") as file:
                content = file.read()
                splited_blocs = content.split("\n\n")

                i: int = 0
                while i < len(splited_blocs):
                    current_bloc = splited_blocs[i].strip()
                    if not current_bloc:
                        i += 1
                        continue

                    if (i + 1 < len(splited_blocs)
                            and current_bloc.startswith("#")
                            and splited_blocs[i + 1].startswith("#")):

                        merged_titles = current_bloc + "\n\n" + splited_blocs[i + 1]
                        fusion_title_list.append(merged_titles)
                        i += 2
                    else:
                        fusion_title_list.append(current_bloc)
                        i += 1

                final_list = self.split_oversized_block(fusion_title_list)

                for bloc in final_list:

                    bloc = bloc.strip()
                    if not bloc:
                        continue
                    is_new_header = bloc.startswith("#")

                    if actual_block and (
                            len(actual_block) + len(bloc) + 2 > 2000
                            or is_new_header):
                        blocks_list.append(actual_block)
                        actual_block = bloc

                    else:
                        if not actual_block:
                            actual_block = bloc
                        else:
                            actual_block += "\n\n" + bloc

                if actual_block:
                    blocks_list.append(actual_block)

                for i, bloc in enumerate(blocks_list):
                    print(f"len bloc {i + 1}: {len(bloc)}")
                    print(bloc)
                    print("\n\n\n")

        except Exception as e:
            print(e)

    def split_oversized_block(self, blocs_list: list[str]):
        """ Split all blocs whose length is greater than 2000.
            Cut them at each \n and return a new list """
        new_list: list[str] = []

        for bloc in blocs_list:
            if len(bloc) > 2000:
                sub_bloc_list = bloc.split("\n")
                new_list.extend(sub_bloc_list)
            else:
                new_list.append(bloc)

        return new_list
