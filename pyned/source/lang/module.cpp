#include <pyned/pyned.h>
#include <pyned/lang/methods.h>

static struct PyMethodDef LangMethods[] =
{
    {
        "parse_file",
        (PyCFunction)parse_file,
        METH_FASTCALL,
        "parses a *.nn file into an AST"
    }
};

static struct PyModuleDef CoreModule =
{
    PyModuleDef_HEAD_INIT,
    "pyned.lang",           // Module name
    NULL,                   // Module documentation - provided in python wrapper
    0,                      // Module does not have any global state
    LangMethods
};

PyMODINIT_FUNC PyInit_lang()
{
    PyObject* pModule = PyModule_Create(&CoreModule);
    if (!pModule)
        return NULL;
    return pModule;
}
