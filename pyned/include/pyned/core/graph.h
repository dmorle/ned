#ifndef PYNED_CORE_GRAPH_H
#define PYNED_CORE_GRAPH_H

#include <pyned/pyned.h>

struct GraphObject
{
    PyObject_HEAD
    core::Graph* pGraph;
};

#ifdef PYNEDC_SRC

extern "C" PyObject* GraphObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
extern "C" int       GraphObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
extern "C" void      GraphObjectDealloc(PyObject* self);

extern "C" PyObject* GraphObjectIsValid    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* GraphObjectOutputSize (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* GraphObjectInputSize  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* GraphObjectListInputs (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* GraphObjectGetOutput  (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* GraphObjectGetInput   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);

static PyMethodDef GraphObjectMethods[] =
{
    {
        "is_valid",
        (PyCFunction)GraphObjectIsValid,
        METH_FASTCALL,
        NULL
    },
    {
        "output_size",
        (PyCFunction)GraphObjectOutputSize,
        METH_FASTCALL,
        NULL
    },
    {
        "input_size",
        (PyCFunction)GraphObjectInputSize,
        METH_FASTCALL,
        NULL
    },
    {
        "list_inputs",
        (PyCFunction)GraphObjectListInputs,
        METH_FASTCALL,
        NULL
    },
    {
        "get_output",
        (PyCFunction)GraphObjectGetOutput,
        METH_FASTCALL,
        NULL
    },
    {
        "get_input",
        (PyCFunction)GraphObjectGetInput,
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

static PyTypeObject GraphObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_pyned.core.Graph",
    .tp_basicsize = sizeof(GraphObject),
    .tp_itemsize = 0,
    .tp_dealloc = GraphObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::core::Graph",
    .tp_methods = GraphObjectMethods,
    .tp_init = GraphObjectInit,
    .tp_new = GraphObjectNew
};

#endif  // PYNEDL_SRC

#endif
