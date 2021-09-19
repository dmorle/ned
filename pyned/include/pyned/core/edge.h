#ifndef PYNED_CORE_EDGE_H
#define PYNED_CORE_EDGE_H

#include <pyned/pyned.h>

struct EdgeObject
{
    PyObject_HEAD
    core::Edge* pEdge;
    bool is_borrowed;
};

#ifdef PYNEDC_SRC

extern "C" PyObject* EdgeObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
extern "C" int       EdgeObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
extern "C" void      EdgeObjectDealloc(PyObject* self);

extern "C" PyObject* EdgeObjectIsValid    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectIsBorrowed (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectHasInput   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectGetInput   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectOutputSize (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectGetOutput  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectGetFWidth  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectGetRank    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* EdgeObjectGetShape   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);

static PyMethodDef EdgeObjectMethods[] =
{
    {
        "is_valid",
        (PyCFunction)EdgeObjectIsValid,
        METH_FASTCALL,
        NULL
    },
    {
        "is_borrowed",
        (PyCFunction)EdgeObjectIsBorrowed,
        METH_FASTCALL,
        NULL
    },
    {
        "has_input",
        (PyCFunction)EdgeObjectHasInput,
        METH_FASTCALL,
        NULL
    },
    {
        "get_input",
        (PyCFunction)EdgeObjectGetInput,
        METH_FASTCALL,
        NULL
    },
    {
        "output_size",
        (PyCFunction)EdgeObjectOutputSize,
        METH_FASTCALL,
        NULL
    },
    {
        "get_output",
        (PyCFunction)EdgeObjectGetOutput,
        METH_FASTCALL,
        NULL
    },
    {
        "get_fwidth",
        (PyCFunction)EdgeObjectGetFWidth,
        METH_FASTCALL,
        NULL
    },
    {
        "get_rank",
        (PyCFunction)EdgeObjectGetRank,
        METH_FASTCALL,
        NULL
    },
    {
        "get_shape",
        (PyCFunction)EdgeObjectGetShape,
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

static PyTypeObject EdgeObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_pyned.core.Edge",
    .tp_basicsize = sizeof(EdgeObject),
    .tp_itemsize = 0,
    .tp_dealloc = EdgeObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::core::Edge",
    .tp_methods = EdgeObjectMethods,
    .tp_init = EdgeObjectInit,
    .tp_new = EdgeObjectNew
};

#endif  // PYNEDL_SRC

#endif
