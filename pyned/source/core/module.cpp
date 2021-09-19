#define PYNEDC_SRC

#include <pyned/pyned.h>
#include <pyned/core/graph.h>
#include <pyned/core/edge.h>
#include <pyned/core/node.h>

static struct PyModuleDef CoreModule =
{
    PyModuleDef_HEAD_INIT,
    "core",                 // Module name
    NULL,                   // Module documentation - provided in python wrapper
    -1                      // Module does not support sub-interpreters
};

PyMODINIT_FUNC PyInit_core()
{
	if (PyType_Ready(&GraphObjectType) < 0)
		return NULL;

	if (PyType_Ready(&EdgeObjectType) < 0)
		return NULL;

	if (PyType_Ready(&NodeObjectType) < 0)
		return NULL;

    PyObject* pModule = PyModule_Create(&CoreModule);
    if (!pModule)
        return NULL;

	Py_INCREF(&GraphObjectType);
	if (PyModule_AddObject(pModule, "Graph", (PyObject*)&GraphObjectType) < 0)
		goto GraphObjectError;

	Py_INCREF(&EdgeObjectType);
	if (PyModule_AddObject(pModule, "Edge", (PyObject*)&EdgeObjectType) < 0)
		goto EdgeObjectError;

	Py_INCREF(&NodeObjectType);
	if (PyModule_AddObject(pModule, "Node", (PyObject*)&NodeObjectType) < 0)
		goto NodeObjectError;

    return pModule;

NodeObjectError:
	Py_DECREF(&NodeObjectType);

EdgeObjectError:
	Py_DECREF(&EdgeObjectType);

GraphObjectError:
	Py_DECREF(&GraphObjectType);

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
	Py_DECREF(pModule);

	PyMem_RawFree(program);
	return 0;
}
