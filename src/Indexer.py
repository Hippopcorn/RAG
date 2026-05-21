from pydantic import BaseModel
from pathlib import Path
import ast


class Chunk(BaseModel):
    """ An object that has an id, store a text, the first index and
        the last index and the file path """
    id: int
    text: str
    file_path: str
    first_index: int
    last_index: int


class Indexer(BaseModel):
    """ Handles retrieving all the interest files from the vllm directotory
        and index them into Chunks """
    dir_path: Path
    chunks_list: list[Chunk] = []
    py_files_paths: list[str] = []
    md_files_paths: list[str] = []

    def get_interest_paths(self):
        """ Get a filtered list with the paths of the .py and .md files
            located in the dir_path """

        for py_file_path in self.dir_path.rglob("*.py"):

            if ("tests" in py_file_path.parts
                    or "__pycache__" in py_file_path.parts):
                continue

            self.py_files_paths.append(str(py_file_path))

        for md_file_path in self.dir_path.rglob("*.md"):

            if ("tests" in md_file_path.parts or "__pycache__"
                    in md_file_path.parts or "pytest_cache"
                    in md_file_path.parts):
                continue

            self.md_files_paths.append(str(md_file_path))

    def index_md_file(self, path: str):
        """ Read a md file, then split it at each \n\n, check alls splits and
            if there are two tittles following, merge them. Call the
            split_oversized_block function to recut splitted_blocs bigger than
            2000 characters. Then, create blocks with titles and text, until
            2000 characters """
        actual_block: str = ""
        blocks_list: list[str] = []
        fusion_title_list: list[str] = []

        try:
            with open(path, "r", encoding="utf-8") as file:
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

                search_start_position = 0
                for bloc in blocks_list:
                    search_start_position = self.create_chunk(
                        bloc, content, path, search_start_position)

        except Exception as e:
            print(e)

    def index_py_file(self, path: str):
        """ Reads a .py file, parses its syntax with AST,
            and creates chunks based on classes and global functions """
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                tree = ast.parse(content, filename=path)
                lines = content.splitlines()

                search_start_position = 0

                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start_line = node.lineno - 1
                        end_line = node.end_lineno

                        bloc_text = "\n".join(lines[start_line:end_line])

                        sub_blocks = self.split_oversized_block([bloc_text])

                        for sub_bloc in sub_blocks:
                            search_start_position = self.create_chunk(
                                sub_bloc, content, path, search_start_position
                            )

                    else:
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        bloc_text = "\n".join(lines[start_line:end_line])

                        if bloc_text.strip():
                            search_start_position = self.create_chunk(
                                bloc_text, content, path, search_start_position
                            )
        except Exception as e:
            print(e)






































    def split_oversized_block(self, blocs_list: list[str]):
        """ Split all blocs whose length is greater than 2000.
            Cut them at each \n and return a new list.
            Re-accumulate their lines safely under 2000 characters """
        new_list: list[str] = []

        for bloc in blocs_list:
            if len(bloc) > 2000:
                lines = bloc.split("\n")
                current_sub_bloc = ""

                for line in lines:
                    if len(line) > 2000:
                        if current_sub_bloc:
                            new_list.append(current_sub_bloc.strip())
                            current_sub_bloc = ""
                        for i in range(0, len(line), 2000):
                            new_list.append(line[i:i+2000])
                        continue

                    if (current_sub_bloc and len(current_sub_bloc)
                            + len(line) + 1 > 2000):
                        new_list.append(current_sub_bloc.strip())
                        current_sub_bloc = line
                    else:
                        if not current_sub_bloc:
                            current_sub_bloc = line
                        else:
                            current_sub_bloc += "\n" + line

                if current_sub_bloc:
                    new_list.append(current_sub_bloc.strip())
            else:
                new_list.append(bloc)

        return new_list

    def create_chunk(self, text: str, content_file: str,
                     file_path: str, search_start_position: int):
        """ Create a chunk with a bloc of text, and calcul it first and last
            index. Add it in the chunks_list """
        first_idx = content_file.find(text, search_start_position)

        if first_idx == -1:
            first_idx = content_file.find(text)

        last_idx = first_idx + len(text)

        chunk = Chunk(
                        id=len(self.chunks_list),
                        text=text,
                        file_path=file_path,
                        first_index=first_idx,
                        last_index=last_idx
                        )
        self.chunks_list.append(chunk)

        return last_idx

    def process_files(self):
        for file_path in self.md_files_paths:
            self.index_md_file(file_path)
