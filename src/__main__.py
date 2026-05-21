from .Indexer import Indexer
from pathlib import Path


def main():
    try:
        indexer = Indexer(dir_path=Path("vllm-0.10.1"))
        indexer.get_interest_paths()
        indexer.process_files()

        for chunk in indexer.chunks_list:
            print(chunk)
            print("\n\n\n")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
