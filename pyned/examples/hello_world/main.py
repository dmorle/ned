from pyned.cpp import lang


def main():
    with open("hello_world.nn", "r") as f:
        ast = lang.parse_file(f)


if __name__ == "__main__":
    main()
