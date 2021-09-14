#ifndef PYNED_LANG_AST_H
#define PYNED_LANG_AST_H

#include <pyned/pyned.h>

struct AstObject
{
    PyObject_HEAD
    lang::AstModule* pAst;
};

extern "C" PyObject* AstObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
extern "C" int       AstObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
extern "C" void      AstObjectDealloc(PyObject* self);

extern "C" PyObject* AstObjectIsValid   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* AstObjectListDefs  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* AstObjectListIntrs (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* AstObjectListFns   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);

static PyMethodDef AstObjectMethods[] =
{
    {
        "is_valid",
        (PyCFunction)AstObjectIsValid,
        METH_FASTCALL,
        NULL
    },
    {
        "list_defs",
        (PyCFunction)AstObjectListDefs,
        METH_FASTCALL,
        NULL
    },
    {
        "list_intrs",
        (PyCFunction)AstObjectListIntrs,
        METH_FASTCALL,
        NULL
    },
    {
        "list_fns",
        (PyCFunction)AstObjectListFns,
        METH_FASTCALL,
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
    .tp_name = "pyned.cpp.lang.ast",
    .tp_basicsize = sizeof(AstObject),
    .tp_itemsize = 0,
    .tp_dealloc = AstObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::lang::AstModule",
    .tp_methods = AstObjectMethods,
    .tp_init = AstObjectInit,
    .tp_new = AstObjectNew
};

#endif
