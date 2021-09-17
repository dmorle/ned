#include <pyned/pyned.h>
#include <pyned/lang/ast.h>

#define pSelf ((AstObject*)self)

/*
 *
 * AstObject Standard Functions
 *
 */

extern "C" PyObject* AstObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyObject* self = type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    pSelf->pAst = NULL;
    return self;
}

extern "C" int AstObjectInit(PyObject* self, PyObject* args, PyObject* kwds)
{
    // Ignoring all arguments
    return 0;
}

extern "C" void AstObjectDealloc(PyObject* self)
{
    if (pSelf->pAst)
        delete pSelf->pAst;

    PyTypeObject* tp = Py_TYPE(self);
    if (tp->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(tp);
    tp->tp_free(self);
}

/*
 *
 * AstObject Member Functions
 *
 */

extern "C" PyObject* AstObjectIsValid(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (pSelf->pAst)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* AstObjectListDefs(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pSelf->pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pSelf->pAst->defs.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pSelf->pAst->defs.size(); i++)
    {
        const std::string& name = pSelf->pAst->defs[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}

extern "C" PyObject * AstObjectListIntrs(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pSelf->pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pSelf->pAst->intrs.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pSelf->pAst->intrs.size(); i++)
    {
        const std::string& name = pSelf->pAst->intrs[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}

extern "C" PyObject * AstObjectListFns(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pSelf->pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pSelf->pAst->fns.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pSelf->pAst->fns.size(); i++)
    {
        const std::string& name = pSelf->pAst->fns[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}
