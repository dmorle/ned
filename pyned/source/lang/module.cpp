#define PYNEDL_SRC

#include <pyned/pyned.h>
#include <pyned/lang/methods.h>
#include <pyned/lang/ast.h>
#include <pyned/lang/obj.h>

#ifndef PYNED_ENV_PTH
#define PYNED_ENV_PTH ""
#endif

static PyMethodDef LangMethods[] =
{
    {
        "parse_file",
        (PyCFunction)parse_file,
        METH_FASTCALL,
        "parses a *.nn file into an AST"
    },
    {
        "eval_ast",
        (PyCFunction)eval_ast,
        METH_FASTCALL,
        "evaluates an AST to produce a computation graph"
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

static PyModuleDef LangModule =
{
    PyModuleDef_HEAD_INIT,
    "lang",                 // Module name
    NULL,                   // Module documentation - provided in python wrapper
    -1,                     // Module does not support sub-interpreters
    LangMethods
};

PyMODINIT_FUNC PyInit_lang()
{
    if (PyType_Ready(&AstObjectType) < 0)
        return NULL;

    if (PyType_Ready(&NedObjObjectType) < 0)
        return NULL;

    PyObject* pModule = PyModule_Create(&LangModule);
    if (!pModule)
        return NULL;

    Py_INCREF(&AstObjectType);
    if (PyModule_AddObject(pModule, "Ast", (PyObject*)&AstObjectType) < 0)
        goto AstObjectError;

    Py_INCREF(&NedObjObjectType);
    if (PyModule_AddObject(pModule, "Obj", (PyObject*)&NedObjObjectType) < 0)
        goto NedObjObjectError;

    return pModule;

NedObjObjectError:
    Py_DECREF(&NedObjObjectType);

AstObjectError:
    Py_DECREF(&AstObjectType);

    Py_DECREF(&pModule);
    return NULL;
}

int main(int argc, char* argv[])
{
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (!program)
    {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    if (PyImport_AppendInittab("_pyned.lang", PyInit_lang) == -1)
    {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    Py_SetProgramName(program);
    Py_Initialize();

    PyObject* pModule = PyImport_ImportModule("_pyned.core");
    if (!pModule)
    {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module '_pyned.core'\n");
    }
    Py_DECREF(pModule);

    PyMem_RawFree(program);

    return 0;
}
