#ifndef PYNED_H
#define PYNED_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <ned/lang/eval.h>
#include <ned/lang/obj.h>
#include <ned/lang/ast.h>
#include <ned/lang/lexer.h>

using namespace nn;

#define PY_SSIZE_T_CLEAN

#define CHECK_ARGNUM(argNum, n)                                            \
	if (argNum != n)                                                       \
	{                                                                      \
		char buf[64];                                                      \
		sprintf(buf, "Expected "#n" argument(s), recieved %llu", argNum);  \
		PyErr_SetString(PyExc_ValueError, buf);                            \
		return NULL;                                                       \
	}

#endif
