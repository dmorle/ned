#define PYNEDC_SRC

#include <pyned/core/graph.h>

#define pSelf ((GraphObject*)self)

/*
 *
 * GraphObject Standard Functions
 *
 */

extern "C" PyObject* GraphObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    PyObject* self = type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    pSelf->pGraph = NULL;
    return self;
}

extern "C" int GraphObjectInit(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // Ignoring all arguments
    pSelf->pGraph = new core::Graph();
    return 0;
}

extern "C" void GraphObjectDealloc(PyObject* self)
{
    if (pSelf->pGraph)
        delete pSelf->pGraph;

    PyTypeObject* tp = Py_TYPE(self);
    if (tp->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(tp);
    tp->tp_free(self);
}
