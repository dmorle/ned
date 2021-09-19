#ifndef PYNED_CORE_NODE_H
#define PYNED_CORE_NODE_H

#include <pyned/pyned.h>

struct NodeObject
{
    PyObject_HEAD
    core::Node* pNode;
    bool is_borrowed;
};

#ifdef PYNEDC_SRC

extern "C" PyObject* NodeObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
extern "C" int       NodeObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
extern "C" void      NodeObjectDealloc(PyObject* self);

extern "C" PyObject* NodeObjectIsValid    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectIsBorrowed (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectGetName    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectGetCargs   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectInputSize  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectGetInput   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectOutputSize (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectGetOutput  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NodeObjectHasOpaque  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);

static PyMethodDef NodeObjectMethods[] =
{
    {
        "is_valid",
        (PyCFunction)NodeObjectIsValid,
        METH_FASTCALL,
        NULL
    },
    {
        "is_borrowed",
        (PyCFunction)NodeObjectIsBorrowed,
        METH_FASTCALL,
        NULL
    },
    {
        "get_cargs",
        (PyCFunction)NodeObjectGetCargs,
        METH_FASTCALL,
        NULL
    },
    {
        "input_size",
        (PyCFunction)NodeObjectInputSize,
        METH_FASTCALL,
        NULL
    },
    {
        "get_input",
        (PyCFunction)NodeObjectGetInput,
        METH_FASTCALL,
        NULL
    },
    {
        "output_size",
        (PyCFunction)NodeObjectOutputSize,
        METH_FASTCALL,
        NULL
    },
    {
        "get_output",
        (PyCFunction)NodeObjectGetOutput,
        METH_FASTCALL,
        NULL
    },
    {
        "has_opaque",
        (PyCFunction)NodeObjectHasOpaque,
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

static PyTypeObject NodeObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_pyned.core.Node",
    .tp_basicsize = sizeof(NodeObject),
    .tp_itemsize = 0,
    .tp_dealloc = NodeObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::core::Node",
    .tp_methods = NodeObjectMethods,
    .tp_init = NodeObjectInit,
    .tp_new = NodeObjectNew
};

#endif  // PYNEDL_SRC

#endif
