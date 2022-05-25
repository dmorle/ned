#define PYNEDC_SRC

#include <pyned/core/node.h>
#include <pyned/core/edge.h>
#include <pyned/lang/obj.h>

#define pSelf ((NodeObject*)self)

/*
 *
 * NodeObject Standard Functions
 *
 */

extern "C" PyObject* NodeObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    PyObject* self = type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    pSelf->pNode = NULL;
    return self;
}

extern "C" int NodeObjectInit(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // Ignoring all arguments
    pSelf->pNode = new (std::nothrow) core::Arg();
    if (!pSelf->pNode)
    {
        PyErr_NoMemory();
        return -1;
    }
    pSelf->is_borrowed = false;
    return 0;
}

extern "C" void NodeObjectDealloc(PyObject* self)
{
    if (pSelf->pNode && !pSelf->is_borrowed)
        delete pSelf->pNode;

    PyTypeObject* tp = Py_TYPE(self);
    if (tp->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(tp);
    tp->tp_free(self);
}

 /*
  *
  * NodeObject Member Functions
  *
  */

extern "C" PyObject* NodeObjectIsValid(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (pSelf->pNode)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* NodeObjectIsBorrowed(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }

    if (pSelf->is_borrowed)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* NodeObjectGetName(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }
    return PyUnicode_FromStringAndSize(pSelf->pNode->name.c_str(), pSelf->pNode->name.size());
}

extern "C" PyObject* NodeObjectGetCargs(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }

    PyObject* pPynedMod = PyImport_ImportModule("_pyned.lang");
    if (!pPynedMod)
        return NULL;
    PyObject* pObjType = PyObject_GetAttrString(pPynedMod, "Obj");
    Py_DECREF(pPynedMod);
    if (!pObjType)
        return NULL;

    PyObject* pList = PyList_New(0);
    if (!pList)
    {
        Py_DECREF(pObjType);
        return PyErr_NoMemory();
    }

    for (auto& e : pSelf->pNode->cargs)
    {
        NedObjObject* pObj = PyObject_New(NedObjObject, (PyTypeObject*)pObjType);
        if (!pObj)
        {
            Py_DECREF(pObjType);
            return PyErr_NoMemory();
        }
        
        // this might cause a segmentation fault because the previous shared_ptr was junk memory (depending on how it was allocated)
        pObj->pObj = e;
        if (PyList_Append(pList, (PyObject*)pObj) == -1)
        {
            Py_DECREF(pObjType);
            return NULL;
        }
    }
    Py_DECREF(pObjType);

    return pList;
}

extern "C" PyObject* NodeObjectInputSize(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(pSelf->pNode->inputs.size());
}

extern "C" PyObject* NodeObjectGetInput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }

    if (!PyLong_Check(args[0]))
    {
        PyErr_SetString(PyExc_TypeError, "Expected int as argument to Node.get_input method");
        return NULL;
    }
    long long idx = PyLong_AsLongLong(args[0]);
    if (idx < 0 or idx >= pSelf->pNode->inputs.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range in Node.get_input method");
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
    pEdge->pEdge = pSelf->pNode->inputs[idx];
    pEdge->is_borrowed = true;

    return (PyObject*)pEdge;
}

extern "C" PyObject* NodeObjectOutputSize(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(pSelf->pNode->outputs.size());
}

extern "C" PyObject* NodeObjectGetOutput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }

    if (!PyLong_Check(args[0]))
    {
        PyErr_SetString(PyExc_TypeError, "Expected int as argument to Node.get_output method");
        return NULL;
    }
    long long idx = PyLong_AsLongLong(args[0]);
    if (idx < 0 or idx >= pSelf->pNode->outputs.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range in Node.get_output method");
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
    pEdge->pEdge = pSelf->pNode->outputs[idx];
    pEdge->is_borrowed = true;

    return (PyObject*)pEdge;
}

extern "C" PyObject* NodeObjectHasOpaque(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pNode)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized node");
        return NULL;
    }

    if (pSelf->pNode->opaque)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}
