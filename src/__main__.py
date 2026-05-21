from .Chunk import Indexer
from pathlib import Path


def main():
    try:
        indexer = Indexer(dir_path=Path("vllm-0.10.1"))
        indexer.get_interest_paths()
        indexer.process_files()

        # indexer.index_md_file("vllm-0.10.1/RELEASE.md")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
