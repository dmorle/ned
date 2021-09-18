#define PYNEDC_SRC

#include <pyned/pyned.h>
#include <pyned/lang/methods.h>

static struct PyModuleDef CoreModule =
{
    PyModuleDef_HEAD_INIT,
    "core",                 // Module name
    NULL,                   // Module documentation - provided in python wrapper
    -1                      // Module does not support sub-interpreters
};

PyMODINIT_FUNC PyInit_core()
{
    PyObject* pModule = PyModule_Create(&CoreModule);
    if (!pModule)
        return NULL;
    return pModule;
}

int main(int argc, char* argv[])
{
	wchar_t* program = Py_DecodeLocale(argv[0], NULL);
	if (!program)
	{
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}

	if (PyImport_AppendInittab("_pyned.core", PyInit_core) == -1)
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

	PyMem_RawFree(program);
	return 0;
}
