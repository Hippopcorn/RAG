from .Indexer import Indexer
from pathlib import Path
from Models import RagDataset


def main():
    try:
        indexer = Indexer(dir_path=Path("vllm-0.10.1"))
        indexer.get_interest_paths()
        indexer.process_files()

        dataset = RagDataset.model_validate_json(
            Path("datasets_public/public/").read_text(encoding="utf-8"))

        # ------------- DEBUG affichage des chunks -----------
        # for chunk in indexer.chunks_list:
        #     print(chunk)
        #     print("\n\n\n")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
