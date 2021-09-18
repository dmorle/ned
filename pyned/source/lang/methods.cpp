#define PYNEDL_SRC

#include <pyned/pyned.h>

#include <pyned/lang/ast.h>
#include <pyned/lang/obj.h>
#include <pyned/core/graph.h>

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
    if (nargs < 2)
    {
        PyErr_SetString(PyExc_ValueError, "Expected more than 2 arguments, recieved 0");
        return NULL;
    }

    // Need to declare everything up front cause of the gotos
    PyObject* pPynedLangMod;
    PyObject* pPynedCoreMod;
    PyObject* pAstType;
    PyObject* pNedObjType;
    PyObject* pGraphType;
    int ret;

    std::string entry_point;
    std::vector<std::shared_ptr<lang::Obj>> cargs;
    lang::EvalCtx* pctx;
    GraphObject* pGraph;

    // A lot of type checking
    pPynedLangMod = PyImport_ImportModule("_pyned.lang");
    if (!pPynedLangMod)
        return NULL;
    pAstType = PyObject_GetAttrString(pPynedLangMod, "Ast");
    if (!pAstType)
        goto on_ast_error;
    ret = PyObject_IsInstance(args[0], pAstType);
    Py_DECREF(pAstType);
    if (ret == -1)
        goto on_ast_error;
    if (ret == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Expected _pyned.lang.Ast type as first argument to eval_ast");
        goto on_ast_error;
    }
    if (!PyUnicode_Check(args[1], &PyBaseString))
    {
        PyErr_SetString(PyExc_TypeError, "Expected str type as second argument to eval_ast");
        goto on_ast_error;
    }

    // Building the cargs
    pNedObjType = PyObject_GetAttrString(pPynedLangMod, "Obj");
    Py_DECREF(pPynedLangMod);
    if (!pNedObjType)
        return NULL;
    for (int i = 2; i < nargs; i++)
    {
        ret = PyObject_IsInstance(args[i], pNedObjType);
        if (ret == -1)
            goto on_ned_obj_error;
        if (ret == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Expected _pyned.lang.Ast type as first argument to eval_ast");
            goto on_ned_obj_error;
        }
    }
    Py_DECREF(pNedObjType);

    // Actually evaluating the script
    entry_point = PyUnicode_AS_DATA(args[1]);
    for (int i = 2; i < nargs; i++)
        cargs.push_back(((NedObjObject*)args[i])->pObj);
    try
    {
        pctx = ((AstObject*)args[0])->pAst->eval(entry_point, cargs);
    }
    catch (lang::GenerationError& err)
    {
        // temporary until I implement tracebacks
        return PyUnicode_FromString(err.what());  // In case of error, it will already be returning NULL anyway
    }

    // Creating the return value
    pPynedCoreMod = PyImport_ImportModule("_pyned.core");
    if (!pPynedCoreMod)
        return NULL;
    pGraphType = PyObject_GetAttrString(pPynedCoreMod, "Graph");
    Py_DECREF(pPynedCoreMod);
    if (!pGraphType)
        return NULL;
    pGraph = PyObject_New(GraphObject, (PyTypeObject*)pGraphType);
    pGraph->pGraph = pctx->pgraph;
    pctx->pgraph = NULL;
    return (PyObject*)pGraph;

on_ned_obj_error:
    Py_DECREF(pNedObjType);
    return NULL;
on_ast_error:
    Py_DECREF(pPynedLangMod);
    return NULL;
}
