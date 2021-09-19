#define PYNEDC_SRC

#include <pyned/core/graph.h>
#include <pyned/core/edge.h>

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
    pSelf->pGraph = new (std::nothrow) core::Graph();
    if (!pSelf->pGraph)
    {
        PyErr_NoMemory();
        return -1;
    }
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

/*
 *
 * GraphObject Member Functions
 *
 */

extern "C" PyObject* GraphObjectIsValid(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (pSelf->pGraph)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* GraphObjectOutputSize(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pGraph)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized graph");
        return NULL;
    }
    return PyLong_FromSize_t(pSelf->pGraph->outputs.size());
}

extern "C" PyObject* GraphObjectInputSize(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pGraph)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized graph");
        return NULL;
    }
    return PyLong_FromSize_t(pSelf->pGraph->inputs.size());
}

extern "C" PyObject* GraphObjectListInputs(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pGraph)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized graph");
        return NULL;
    }
    PyObject* pList = PyList_New(0);
    if (!pList)
        return NULL;
    for (auto& [key, val] : pSelf->pGraph->inputs)
    {
        PyObject* pStr = PyUnicode_FromStringAndSize(key.c_str(), key.size());
        if (!pStr)
            return NULL;
        if (PyList_Append(pList, pStr) == -1)
            return NULL;
    }
    return pList;
}

extern "C" PyObject* GraphObjectGetOutput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    if (!pSelf->pGraph)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized graph");
        return NULL;
    }

    if (!PyLong_Check(args[0]))
    {
        PyErr_SetString(PyExc_TypeError, "Expected long as first argument to Graph.get_output method");
        return NULL;
    }
    long long idx = PyLong_AsLongLong(args[0]);
    if (idx < 0 || idx >= pSelf->pGraph->outputs.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range in Graph.get_output method");
        return NULL;
    }

    PyObject* pPynedMod = PyImport_ImportModule("_pyned.core");
    if (!pPynedMod)
        return NULL;
    PyObject* pEdgeType = PyObject_GetAttrString(pPynedMod, "Edge");
    Py_DECREF(pPynedMod);
    if (!pEdgeType)
        return NULL;
    EdgeObject* pEdge = PyObject_New(EdgeObject, (PyTypeObject*)pEdgeType);
    Py_DECREF(pEdgeType);
    if (!pEdge)
        return PyErr_NoMemory();
    pEdge->pEdge = pSelf->pGraph->outputs[idx];
    pEdge->is_borrowed = true;

    return (PyObject*)pEdge;
}

extern "C" PyObject* GraphObjectGetInput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    if (!pSelf->pGraph)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized graph");
        return NULL;
    }

    if (!PyUnicode_Check(args[0]))
    {
        PyErr_SetString(PyExc_TypeError, "Expected str as first argument to Graph.get_input method");
        return NULL;
    }
    PyObject* pInputASCII = PyUnicode_AsASCIIString(args[0]);
    if (!pInputASCII)
        return NULL;
    char* input_name = PyBytes_AsString(pInputASCII);
    Py_DECREF(pInputASCII);
    if (!input_name)
        return NULL;

    if (!pSelf->pGraph->inputs.contains(input_name))
        return PyErr_Format(PyExc_KeyError, "'%s' is not an input to the graph", input_name);
    
    PyObject* pPynedMod = PyImport_ImportModule("_pyned.core");
    if (!pPynedMod)
        return NULL;
    PyObject* pEdgeType = PyObject_GetAttrString(pPynedMod, "Edge");
    Py_DECREF(pPynedMod);
    if (!pEdgeType)
        return NULL;
    EdgeObject* pEdge = PyObject_New(EdgeObject, (PyTypeObject*)pEdgeType);
    Py_DECREF(pEdgeType);
    if (!pEdge)
        return PyErr_NoMemory();
    pEdge->pEdge = pSelf->pGraph->inputs[input_name];
    pEdge->is_borrowed = true;

    return (PyObject*)pEdge;
}
