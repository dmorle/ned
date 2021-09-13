from pyned.cpp.lang import parse_file as cpp_parse_file
from .errors import NedSyntaxError


def parse_file(file):
    with open(file, "r") as f:
        ret = cpp_parse_file(f)
    if type(ret) == tuple:
        raise NedSyntaxError(*ret)
    return ret
