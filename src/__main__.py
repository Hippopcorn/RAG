# from .Models import RagDataset
from .CommandLineInterface import CLI
import fire


def main():
    try:
        fire.Fire(CLI)

        # dataset = RagDataset.model_validate_json(
        #     Path("datasets_public/public/").read_text(encoding="utf-8"))

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
