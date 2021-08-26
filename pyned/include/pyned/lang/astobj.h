#ifndef PYNED_LANG_AST_H
#define PYNED_LANG_AST_H

#include <pyned/pyned.h>

struct AstObject
{
    PyObject_HEAD
    impl::AstModule ast;
};

PyObject* AstObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
int       AstObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
void      AstObjectDealloc(PyObject* self);

PyObject* AstObjectEval(PyObject* self, PyObject* args);

static PyMethodDef AstObjectMethods[] =
{
    {
        "eval",
        AstObjectEval,
        METH_VARARGS,
        NULL
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

static PyTypeObject AstObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyned.lang.ast",
    .tp_basicsize = sizeof(AstObject),
    .tp_itemsize = 0,
    .tp_dealloc = AstObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::impl::AstModule",
    .tp_methods = AstObjectMethods,
    .tp_init = AstObjectInit,
    .tp_new = AstObjectNew
};

#endif
