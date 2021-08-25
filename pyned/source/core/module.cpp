#include <pyned/pyned.h>
#include <pyned/lang/methods.h>

static struct PyModuleDef CoreModule =
{
    PyModuleDef_HEAD_INIT,
    "pyned.core",           // Module name
    NULL,                   // Module documentation - provided in python wrapper
    0                       // Module does not have any global state
};

PyMODINIT_FUNC PyInit_core()
{
    PyObject* pModule = PyModule_Create(&CoreModule);
    if (!pModule)
        return NULL;
    return pModule;
}
