from .Chunk import Indexer


def main():
    try:
        indexer = Indexer()
        interest_files_path: list = indexer.get_interest_paths()
        # print(interest_files_path)

        indexer.index_md_file("vllm-0.10.1/README.md")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
