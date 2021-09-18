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

	PyObject* pPath = PySys_GetObject("path");
	if (!pPath)
	{
		fprintf(stderr, "Error: unable to retrieve the python path variable\n");
		exit(-1);
	}
	if (!PyList_Check(pPath))
	{
		fprintf(stderr, "Error: sys.path is not a list object");
		exit(-1);
	}
	PyObject* pPkgPath = PyUnicode_FromString(PYNED_ENV_PTH);
	PyList_Append(pPath, pPkgPath);

	PyObject* pModule = PyImport_ImportModule("_pyned.lang");
	if (!pModule)
	{
		PyErr_Print();
		fprintf(stderr, "Error: could not import module '_pyned.lang'\n");
	}

	PyObject* dict = PyDict_New();
	PyObject* pPynedLang = PyImport_ImportModule("pyned.lang");
	PyDict_SetItemString(dict, "lang", pPynedLang);
	PyObject* ret = PyRun_String("lang.parse_file(\"C:/Users/dmorl/source/repos/ned/pyned/examples/hello_world/hello_world.nn\")", Py_single_input, dict, dict);
	Py_DECREF(dict);
	if (ret)
		Py_DECREF(ret);
	else
		PyErr_Print();

	// cleaning up the sys.path variable
	PyObject* npPath = PyList_GetSlice(pPath, 0, PyList_GET_SIZE(pPath) - 2);
	Py_DECREF(pPath);
	Py_DECREF(pPkgPath);
	PySys_SetObject("path", npPath);

	PyMem_RawFree(program);

	return 0;
}
