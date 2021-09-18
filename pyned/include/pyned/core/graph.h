#ifndef PYNED_LANG_GRAPH_H
#define PYNED_LANG_GRAPH_H

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

//extern "C" PyObject* GraphObject

static PyMethodDef GraphObjectMethods[] =
{
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
