#ifndef PYNED_LANG_SYNTAXERR_H
#define PYNED_LANG_SYNTAXERR_H

#include <pyned/pyned.h>

struct SyntaxerrObject
{
    PyObject_HEAD
    impl::SyntaxError err;
};

PyObject* SyntaxerrObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
int       SyntaxerrObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
void      SyntaxerrObjectDealloc(PyObject* self);

static PyMethodDef SyntaxerrObjectMethods[] =
{
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

static PyTypeObject SyntaxerrObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyned.lang.Syntaxerr",
    .tp_basicsize = sizeof(SyntaxerrObject),
    .tp_itemsize = 0,
    .tp_dealloc = SyntaxerrObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::impl::SyntaxError",
    .tp_methods = SyntaxerrObjectMethods,
    .tp_init = SyntaxerrObjectInit,
    .tp_new = SyntaxerrObjectNew
};

#endif
