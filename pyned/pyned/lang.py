from _pyned import lang as _lang
from .core import Graph


class NedSyntaxError(Exception):
    def __init__(self, line_num, col_num, err_msg):
        super(NedSyntaxError, self).__init__("line %d - column %d\n%s" % (line_num, col_num, err_msg))

        self.line_num = line_num
        self.col_num = col_num


class NedGenerationError(Exception):
    def __init__(self, err_msg):
        super(NedGenerationError, self).__init__(err_msg)


class Obj:
    def __init__(self, data):
        if type(data) is _lang.Obj:
            self._obj = data
        else:
            self._obj = _lang.Obj(data)


class Ast:
    def __init__(self, ast: _lang.Ast):
        assert type(ast) is _lang.Ast
        assert ast.is_valid()
        self.ast: _lang.Ast = ast

    def eval(self, entry_point: str, *cargs: Obj) -> Graph:
        ret = _lang.eval_ast(self.ast, entry_point, *cargs)
        if type(ret) is str:
            raise NedGenerationError(ret)
        return Graph(ret)


def parse_file(file) -> Ast:
    with open(file, "r") as f:
        ret = _lang.parse_file(f)
    if type(ret) == tuple:
        raise NedSyntaxError(*ret)
    return Ast(ret)
