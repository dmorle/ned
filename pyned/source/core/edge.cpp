#define PYNEDC_SRC

#include <pyned/core/edge.h>
#include <pyned/core/node.h>

#define pSelf ((EdgeObject*)self)

/*
 *
 * EdgeObject Standard Functions
 *
 */

extern "C" PyObject * EdgeObjectNew(PyTypeObject * type, PyObject * args, PyObject * kwargs)
{
    PyObject* self = type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    pSelf->pEdge = NULL;
    return self;
}

extern "C" int EdgeObjectInit(PyObject * self, PyObject * args, PyObject * kwargs)
{
    // Ignoring all arguments
    pSelf->pEdge = new (std::nothrow) core::Edge();
    if (!pSelf->pEdge)
    {
        PyErr_NoMemory();
        return -1;
    }
    pSelf->is_borrowed = false;
    return 0;
}

extern "C" void EdgeObjectDealloc(PyObject * self)
{
    if (pSelf->pEdge && !pSelf->is_borrowed)
        delete pSelf->pEdge;

    PyTypeObject* tp = Py_TYPE(self);
    if (tp->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(tp);
    tp->tp_free(self);
}

/*
 *
 * EdgeObject Member Functions
 *
 */

extern "C" PyObject* EdgeObjectIsValid(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (pSelf->pEdge)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* EdgeObjectIsBorrowed(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }

    if (pSelf->is_borrowed)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject * EdgeObjectHasInput(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }

    if (pSelf->pEdge->input)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* EdgeObjectGetInput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }

    PyObject* pPynedMod = PyImport_ImportModule("_pyned.core");
    if (!pPynedMod)
        return NULL;
    PyObject* pNodeType = PyObject_GetAttrString(pPynedMod, "Node");
    Py_DECREF(pPynedMod);
    if (!pNodeType)
        return NULL;
    NodeObject* pNode = PyObject_New(NodeObject, (PyTypeObject*)pNodeType);
    Py_DECREF(pNodeType);
    if (!pNode)
        return PyErr_NoMemory();
    pNode->pNode = pSelf->pEdge->input;
    pNode->is_borrowed = true;

    PyObject* pId;
    PyObject* pRet;

    pId = PyLong_FromLong(pSelf->pEdge->inpid);
    if (!pId)
        goto id_error;
    pRet = PyTuple_New(2);
    if (!pRet)
        goto tuple_error;
    PyTuple_SET_ITEM(pRet, 0, (PyObject*)pNode);
    PyTuple_SET_ITEM(pRet, 1, pId);

    return pRet;

tuple_error:
    Py_DECREF(pId);

id_error:
    Py_DECREF(pNode);

    return PyErr_NoMemory();
}

extern "C" PyObject* EdgeObjectOutputSize(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(pSelf->pEdge->outputs.size());
}

extern "C" PyObject* EdgeObjectGetOutput(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }

    if (!PyLong_Check(args[0]))
    {
        PyErr_SetString(PyExc_TypeError, "Expected int as argument to Edge.get_output method");
        return NULL;
    }
    long long idx = PyLong_AsLongLong(args[0]);
    if (idx < 0 or idx >= pSelf->pEdge->outputs.size())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range in Edge.get_output method");
        return NULL;
    }

    PyObject* pPynedMod = PyImport_ImportModule("_pyned.core");
    if (!pPynedMod)
        return NULL;
    PyObject* pNodeType = PyObject_GetAttrString(pPynedMod, "Node");
    Py_DECREF(pPynedMod);
    if (!pNodeType)
        return NULL;
    NodeObject* pNode = PyObject_New(NodeObject, (PyTypeObject*)pNodeType);
    Py_DECREF(pNodeType);
    if (!pNode)
        return PyErr_NoMemory();
    pNode->pNode = std::get<0>(pSelf->pEdge->outputs[idx]);
    pNode->is_borrowed = true;

    PyObject* pId;
    PyObject* pRet;

    pId = PyLong_FromLong(std::get<1>(pSelf->pEdge->outputs[idx]));
    if (!pId)
        goto id_error;
    pRet = PyTuple_New(2);
    if (!pRet)
        goto tuple_error;
    PyTuple_SET_ITEM(pRet, 0, (PyObject*)pNode);
    PyTuple_SET_ITEM(pRet, 1, pId);

    return pRet;

tuple_error:
    Py_DECREF(pId);

id_error:
    Py_DECREF(pNode);

    return PyErr_NoMemory();
}

extern "C" PyObject* EdgeObjectGetFWidth(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }
    // 0 => F16
    // 1 => F32
    // 2 => F64
    return PyLong_FromLong((long)pSelf->pEdge->dsc.dty);
}

extern "C" PyObject* EdgeObjectGetRank(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(pSelf->pEdge->dsc.rk);
}

extern "C" PyObject* EdgeObjectGetShape(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);
    if (!pSelf->pEdge)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized edge");
        return NULL;
    }
    
    PyObject* pList = PyList_New(0);
    for (auto e : pSelf->pEdge->dsc.dims)
    {
        PyObject* pDim = PyLong_FromUnsignedLong(e);
        if (!pDim)
        {
            Py_DECREF(pList);
            return PyErr_NoMemory();
        }
        if (PyList_Append(pList, pDim) == -1)
        {
            Py_DECREF(pList);
            Py_DECREF(pDim);
            return NULL;
        }
    }
    return pList;
}
