#ifndef PYNED_LANG_METHODS_H
#define PYNED_LANG_METHODS_H

#include <pyned/pyned.h>

extern "C" PyObject* parse_file(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* eval_ast(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

#endif
