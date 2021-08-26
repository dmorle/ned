#ifndef PYNED_LANG_GENERR_H
#define PYNED_LANG_GENERR_H

#include <pyned/pyned.h>

struct GenerrObject
{
    PyObject_HEAD
    impl::GenerationError err;
};

PyObject* GenerrObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
int       GenerrObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
void      GenerrObjectDealloc(PyObject* self);

static PyMethodDef GenerrObjectMethods[] =
{
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

static PyTypeObject GenerrObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyned.lang.Generr",
    .tp_basicsize = sizeof(GenerrObject),
    .tp_itemsize = 0,
    .tp_dealloc = GenerrObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::impl::GenerationError",
    .tp_methods = GenerrObjectMethods,
    .tp_init = GenerrObjectInit,
    .tp_new = GenerrObjectNew
};

#endif
