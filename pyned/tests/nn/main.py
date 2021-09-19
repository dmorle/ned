from pyned import lang


def main():
    ast = lang.parse_file("hello_world.nn")
    lang.eval_ast(ast, "model")


if __name__ == "__main__":
    main()
