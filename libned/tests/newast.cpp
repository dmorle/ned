#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>

#include <ned/errors.h>
#include <ned/lang/ast.h>

using namespace nn;
using namespace lang;

bool test_ast() {
    TokenArray tarr;
    std::string fpth = TESTS_DIR "newast.nn";
    if (lex_file(fpth.c_str(), tarr))
        return true;

    AstModule ast;
    if (parse_module(tarr, ast))
        return true;

    return false;
}

int main() {
    if (test_ast()) {
        error::print();
    }
    return 0;
}
