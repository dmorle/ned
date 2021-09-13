#include <pyned/pyned.h>
#include <pyned/lang/ast.h>

#define pSelf   ((AstObject*)self)
#define pAst   (pSelf->pAst)

/*
 *
 * AstObject Standard Functions
 *
 */

extern "C" PyObject* AstObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    AstObject* self = (AstObject*)type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    return (PyObject*)self;
}

extern "C" int AstObjectInit(PyObject* self, PyObject* args, PyObject* kwds)
{
    // Ignoring all arguments
    pAst = NULL;
    return 0;
}

extern "C" void AstObjectDealloc(PyObject* self)
{
    PyTypeObject* tp = Py_TYPE(self);

    if (pAst)
        delete pAst;

    tp->tp_free(self);
    Py_DECREF(tp);
}

/*
 *
 * AstObject Member Functions
 *
 */

extern "C" PyObject* AstObjectIsValid(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (pAst)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C" PyObject* AstObjectListDefs(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pAst->defs.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pAst->defs.size(); i++)
    {
        const std::string& name = pAst->defs[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}

extern "C" PyObject * AstObjectListIntrs(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pAst->intrs.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pAst->intrs.size(); i++)
    {
        const std::string& name = pAst->intrs[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}

extern "C" PyObject * AstObjectListFns(PyObject * self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 0);

    if (!pAst)
    {
        PyErr_SetString(PyExc_ValueError, "Uninitialized ast object");
        return NULL;
    }

    PyObject* pList = PyList_New(pAst->fns.size());
    if (!pList)
        return PyErr_NoMemory();
    for (size_t i = 0; i < pAst->fns.size(); i++)
    {
        const std::string& name = pAst->fns[i].get_name();
        PyObject* pStr = PyUnicode_FromStringAndSize(name.c_str(), name.size());
        // TODO: check for memory allocation failure
        PyList_SET_ITEM(pList, i, pStr);
    }
    return pList;
}
