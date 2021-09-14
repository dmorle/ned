#ifndef PYNED_LANG_OBJS_H
#define PYNED_LANG_OBJS_H

#include <pyned/pyned.h>
#include <memory>

struct NedObjObject
{
    PyObject_HEAD
    std::shared_ptr<lang::Obj> pObj;
};

extern "C" PyObject* NedObjObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs);
extern "C" int       NedObjObjectInit(PyObject* self, PyObject* args, PyObject* kwargs);
extern "C" void      NedObjObjectDealloc(PyObject* self);

extern "C" PyObject* NedObjObjectGet    (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NedObjObjectInst   (PyObject* self, PyObject* const* args, Py_ssize_t nargs);
extern "C" PyObject* NedObjObjectGetType(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

static PyMethodDef NedObjObjectMethods[] =
{
    {
        "get",
        (PyCFunction)NedObjObjectGet,
        METH_FASTCALL,
        NULL
    },
    {
        "instantiate",
        (PyCFunction)NedObjObjectInst,
        METH_FASTCALL,
        NULL
    },
    {
        "get_type",
        (PyCFunction)NedObjObjectGetType,
        METH_FASTCALL,
        NULL
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

static PyTypeObject NedObjObjectType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyned.cpp.lang.obj",
    .tp_basicsize = sizeof(NedObjObject),
    .tp_itemsize = 0,
    .tp_dealloc = NedObjObjectDealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "raw python wrapper for nn::lang::Obj",
    .tp_methods = NedObjObjectMethods,
    .tp_init = NedObjObjectInit,
    .tp_new = NedObjObjectNew
};

#endif
