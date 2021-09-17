#include <pyned/pyned.h>

#include <pyned/lang/ast.h>
#include <pyned/lang/ast.h>

#include <csignal>

extern "C" PyObject* parse_file(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    int fd = PyObject_AsFileDescriptor(args[0]);
    if (fd == -1)
        return NULL;

    PyObject* pPynedMod = PyImport_ImportModule("_pyned.lang");
    if (!pPynedMod)
        return NULL;
    PyObject* pAstType = PyObject_GetAttrString(pPynedMod, "Ast");
    Py_DECREF(pPynedMod);
    if (!pAstType)
        return NULL;
    AstObject* pAst = PyObject_New(AstObject, (PyTypeObject*)pAstType);
    Py_DECREF(pAstType);
    if (!pAst)
        return PyErr_NoMemory();
    pAst->pAst = NULL;

#ifdef WIN32
    fd = _dup(fd);
    if (fd == -1)
    {
        Py_DecRef((PyObject*)pAst);
        return PyErr_SetFromErrno(PyExc_OSError);
    }

    FILE* pf = _fdopen(fd, "rb");
#else
    FILE* pf = fdopen(dup(fd), "rb");
#endif

    lang::TokenArray tarr;
    try
    {
        lang::lex_file(pf, tarr);
        pAst->pAst = new (std::nothrow) lang::AstModule(tarr);
    }
    catch (lang::SyntaxError& err)
    {

        fclose(pf);
        Py_DecRef((PyObject*)pAst);
        
        PyObject* pRet;
        PyObject* pLineNum;
        PyObject* pColNum;
        PyObject* pErrMsg;

        pRet = PyTuple_New(3);
        if (!pRet)
            goto TupleMemoryError;
        pLineNum = PyLong_FromUnsignedLongLong((unsigned long long)err.line_num);
        if (!pLineNum)
            goto LineNumMemoryError;
        pColNum = PyLong_FromUnsignedLongLong((unsigned long long)err.col_num);
        if (!pColNum)
            goto ColNumMemoryError;
        pErrMsg = PyUnicode_FromStringAndSize(err.errmsg.c_str(), (Py_ssize_t)err.errmsg.size());
        if (!pErrMsg)
            goto ErrMsgMemoryError;

        PyTuple_SET_ITEM(pRet, 0, pLineNum);
        PyTuple_SET_ITEM(pRet, 1, pColNum);
        PyTuple_SET_ITEM(pRet, 2, pErrMsg);

        return pRet;

    ErrMsgMemoryError:
        Py_DecRef(pColNum);
    ColNumMemoryError:
        Py_DecRef(pLineNum);
    LineNumMemoryError:
        Py_DecRef(pRet);
    TupleMemoryError:
        return PyErr_NoMemory();
    }
    
    if (fclose(pf) == EOF)
        return PyErr_SetFromErrno(PyExc_OSError);

    if (!pAst->pAst)
    {
        Py_DecRef((PyObject*)pAst);
        return PyErr_NoMemory();
    }

    return (PyObject*)pAst;
}

extern "C" PyObject* eval_ast(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    return NULL;
}
