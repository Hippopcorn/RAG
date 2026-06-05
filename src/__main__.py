from .CommandLineInterface import CLI
import fire


def main():
    try:
        fire.Fire(CLI)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
