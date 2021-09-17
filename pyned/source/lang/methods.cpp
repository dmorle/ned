#include <pyned/pyned.h>

#include <pyned/lang/ast.h>
#include <pyned/lang/ast.h>

extern "C" PyObject* parse_file(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(nargs, 1);
    int fd = PyObject_AsFileDescriptor(args[0]);
    if (fd == -1)
        return NULL;

    AstObject* pAst = PyObject_New(AstObject, &AstObjectType);
    if (!pAst)
        return PyErr_NoMemory();

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
        pLineNum = PyLong_FromUnsignedLong(err.line_num);
        if (!pLineNum)
            goto LineNumMemoryError;
        pColNum = PyLong_FromUnsignedLong(err.col_num);
        if (!pColNum)
            goto ColNumMemoryError;
        pErrMsg = PyUnicode_FromStringAndSize(err.errmsg.c_str(), err.errmsg.size());
        if (!pErrMsg)
            goto ErrMsgMemoryError;
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
    fclose(pf);
    
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
