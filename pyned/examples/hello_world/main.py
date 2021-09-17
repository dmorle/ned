from pyned import lang


def main():
    ast = lang.parse_file("pyned/examples/hello_world/hello_world.nn")
    print(ast)


if __name__ == "__main__":
    main()
