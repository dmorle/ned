#include <pyned/pyned.h>
#include <pyned/lang/methods.h>

static struct PyMethodDef LangMethods[] =
{
    {
        "parse_file",
        (PyCFunction)parse_file,
        METH_FASTCALL,
        "parses a *.nn file into an AST"
    },
	{
		NULL,
		NULL,
		0,
		NULL
	}
};

static struct PyModuleDef LangModule =
{
    PyModuleDef_HEAD_INIT,
    "lang",                 // Module name
    NULL,                   // Module documentation - provided in python wrapper
    -1,                     // Module does not support sub-interpreters
    LangMethods
};

PyMODINIT_FUNC PyInit_lang()
{
    PyObject* pModule = PyModule_Create(&LangModule);
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

	if (PyImport_AppendInittab("pyned.lang", PyInit_lang) == -1)
	{
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}

	Py_SetProgramName(program);
	Py_Initialize();

	PyObject* pModule = PyImport_ImportModule("pyned.lang");
	if (!pModule)
	{
		PyErr_Print();
		fprintf(stderr, "Error: could not import module 'pyned.lang'\n");
	}

	PyMem_RawFree(program);
	return 0;
}
