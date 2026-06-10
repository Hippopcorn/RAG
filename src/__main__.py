from .CommandLineInterface import CLI
from pydantic import ValidationError
from json import JSONDecodeError
import fire


def main() -> None:
    try:
        fire.Fire(CLI)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
    except (JSONDecodeError, ValidationError) as e:
        print(f"Error: invalid input file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
