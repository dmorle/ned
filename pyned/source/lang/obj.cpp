#define PYNEDL_SRC

#include <pyned/pyned.h>
#include <pyned/lang/obj.h>

#define pSelf   ((NedObjObject*)self)
#define m_obj   (pSelf->pObj)

extern "C" std::shared_ptr<lang::Obj> pyToNed(PyObject * pVal)
{
    if (PyUnicode_Check(pVal))
    {
        PyObject* pAsciiStr = PyUnicode_AsASCIIString(pVal);
        char* str = PyBytes_AsString(pAsciiStr);

        if (strcmp(str, "dtype.f16") == 0)
            return lang::create_obj_fwidth(core::tensor_dty::F16);
        else if (strcmp(str, "dtype.f32") == 0)
            return lang::create_obj_fwidth(core::tensor_dty::F32);
        else if (strcmp(str, "dtype.f64") == 0)
            return lang::create_obj_fwidth(core::tensor_dty::F64);
        else if (strcmp(str, "fwidth") == 0)
            return lang::create_obj_dtype(lang::ObjType::FWIDTH);
        else if (strcmp(str, "bool") == 0)
            return lang::create_obj_dtype(lang::ObjType::BOOL);
        else if (strcmp(str, "int") == 0)
            return lang::create_obj_dtype(lang::ObjType::INT);
        else if (strcmp(str, "float") == 0)
            return lang::create_obj_dtype(lang::ObjType::FLOAT);
        else if (strcmp(str, "str") == 0)
            return lang::create_obj_dtype(lang::ObjType::STR);
        else if (strcmp(str, "array") == 0)
            return lang::create_obj_dtype(lang::ObjType::ARRAY);
        else if (strcmp(str, "tuple") == 0)
            return lang::create_obj_dtype(lang::ObjType::TUPLE);
        else if (strcmp(str, "tensor") == 0)
            return lang::create_obj_dtype(lang::ObjType::TENSOR);
        else
            return lang::create_obj_str(str);
    }
    else if (PyBool_Check(pVal))
    {
        int ret = PyObject_IsTrue(pVal);
        if (ret < 0)
        {
            PyErr_SetString(PyExc_ValueError, "Unable to retrieve truth value from boolean");
            return nullptr;
        }
        return lang::create_obj_bool(ret);
    }
    else if (PyLong_Check(pVal))
    {
        long ret = PyLong_AsLong(pVal);
        if (ret == -1 && !PyErr_Occurred())
        {
            PyErr_SetString(PyExc_ValueError, "Unable to retrieve value from int");
            return nullptr;
        }
        return lang::create_obj_int(ret);
    }
    else if (PyFloat_Check(pVal))
    {
        float ret = PyFloat_AsDouble(pVal);
        if (ret == -1 && !PyErr_Occurred())
        {
            PyErr_SetString(PyExc_ValueError, "Unable to retrieve value from float");
            return nullptr;
        }
        return lang::create_obj_float(ret);
    }
    else if (PyList_Check(pVal))
    {
        std::vector<std::shared_ptr<lang::Obj>> objs;
        Py_ssize_t len = PyList_GET_SIZE(pVal);
        for (Py_ssize_t i = 0; i < len; i++)
        {
            objs.push_back(pyToNed(PyList_GET_ITEM(pVal, i)));
            if (!objs.back())
                return nullptr;
        }
        return lang::create_obj_tuple(objs);
    }
    else if (PyTuple_Check(pVal))
    {
        std::vector<std::shared_ptr<lang::Obj>> objs;
        Py_ssize_t len = PyTuple_GET_SIZE(pVal);
        for (Py_ssize_t i = 0; i < len; i++)
        {
            objs.push_back(pyToNed(PyTuple_GET_ITEM(pVal, i)));
            if (!objs.back())
                return nullptr;
        }
        return lang::create_obj_tuple(objs);
    }
    PyErr_SetString(PyExc_TypeError, "Invalid initialization type for ned object");
    return nullptr;
}

extern "C" PyObject* nedToPy(const std::shared_ptr<lang::Obj>& obj)
{
    PyObject* pret;
    switch (obj->ty)
    {
    case lang::ObjType::TYPE:
        PyErr_SetString(PyExc_ValueError, "Ned type object does not have a python mapping");
        return NULL;
    case lang::ObjType::INVALID:
        PyErr_SetString(PyExc_ValueError, "Ned invalid object does not have a python mapping");
        return NULL;
    case lang::ObjType::VAR:
        PyErr_SetString(PyExc_ValueError, "Ned var object does not have a python mapping");
        return NULL;
    case lang::ObjType::FWIDTH:
        PyErr_SetString(PyExc_ValueError, "Ned fwidth object does not have a python mapping");
        return NULL;
    case lang::ObjType::BOOL:
        pret = PyBool_FromLong(std::static_pointer_cast<lang::ObjBool>(obj)->data.val);
        break;
    case lang::ObjType::INT:
        pret = PyLong_FromLongLong(std::static_pointer_cast<lang::ObjInt>(obj)->data.val);
        break;
    case lang::ObjType::FLOAT:
        pret = PyFloat_FromDouble(std::static_pointer_cast<lang::ObjFloat>(obj)->data.val);
        break;
    case lang::ObjType::STR:
        pret = PyUnicode_FromStringAndSize(
            std::static_pointer_cast<lang::ObjStr>(obj)->data.val.c_str(),
            std::static_pointer_cast<lang::ObjStr>(obj)->data.val.size()
        );
        break;
    case lang::ObjType::ARRAY:
        pret = PyList_New(std::static_pointer_cast<lang::ObjArray>(obj)->data.elems.size());
        for (Py_ssize_t i = 0; i < std::static_pointer_cast<lang::ObjArray>(obj)->data.elems.size(); i++)
        {
            PyObject* pelem = nedToPy(std::static_pointer_cast<lang::ObjArray>(obj)->data.elems[i]);
            if (!pelem)
            {
                Py_DECREF(pret);  // assuming this cleans up the already assigned elements
                return NULL;
            }
            PyList_SET_ITEM(pret, i, pelem);
        }
        break;
    case lang::ObjType::TUPLE:
        pret = PyTuple_New(std::static_pointer_cast<lang::ObjTuple>(obj)->data.elems.size());
        for (Py_ssize_t i = 0; i < std::static_pointer_cast<lang::ObjTuple>(obj)->data.elems.size(); i++)
        {
            PyObject* pelem = nedToPy(std::static_pointer_cast<lang::ObjTuple>(obj)->data.elems[i]);
            if (!pelem)
            {
                Py_DECREF(pret);  // assuming this cleans up the already assigned elements
                return NULL;
            }
            PyTuple_SET_ITEM(pret, i, pelem);
        }
        break;
    case lang::ObjType::TENSOR:
        PyErr_SetString(PyExc_ValueError, "Ned tensor object does not have a python mapping");
        return NULL;
    case lang::ObjType::DEF:
        PyErr_SetString(PyExc_ValueError, "Ned def object does not have a python mapping");
        return NULL;
    case lang::ObjType::FN:
        PyErr_SetString(PyExc_ValueError, "Ned fn object does not have a python mapping");
        return NULL;
    case lang::ObjType::INTR:
        PyErr_SetString(PyExc_ValueError, "Ned intr object does not have a python mapping");
        return NULL;
    case lang::ObjType::MODULE:
        PyErr_SetString(PyExc_ValueError, "Ned module object does not have a python mapping");
        return NULL;
    case lang::ObjType::PACKAGE:
        PyErr_SetString(PyExc_ValueError, "Ned package object does not have a python mapping");
        return NULL;
    default:
        PyErr_SetString(PyExc_ValueError, "Unrecognized ned object type");
        return NULL;
    }
    return pret;
}

/*
 *
 * NedObjObject Standard Functions
 *
 */

extern "C" PyObject* NedObjObjectNew(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    PyObject* self = type->tp_alloc(type, 0);
    if (!self)
        return PyErr_NoMemory();
    m_obj = nullptr;
    return self;
}

extern "C" int NedObjObjectInit(PyObject* self, PyObject* args, PyObject* kwargs)
{
    if (PyTuple_Size(args) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "Expected 1 argument");
        return -1;
    }

    PyObject* pVal = PyTuple_GetItem(args, 0);
    m_obj = pyToNed(pVal);
    if (!m_obj)
        return -1;
    return 0;
}

extern "C" void NedObjObjectDealloc(PyObject* self)
{
    PyTypeObject* tp = Py_TYPE(self);
    if (tp->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_DECREF(tp);
    tp->tp_free(self);
}

/*
 *
 * NedObjObject Member Functions
 *
 */

extern "C" PyObject* NedObjObjectGet(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    return nedToPy(m_obj);
}

extern "C" PyObject* NedObjObjectGetInst(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    std::shared_ptr<lang::Obj> inst;
    try
    {
        inst = m_obj->inst();
    }
    catch (lang::GenerationError& generr)
    {
        PyErr_SetString(PyExc_ValueError, generr.what());
        return NULL;
    }
    return nedToPy(inst);
}

extern "C" PyObject* NedObjObjectGetType(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    std::shared_ptr<lang::Obj> type;
    try
    {
        type = m_obj->type();
    }
    catch (lang::GenerationError& generr)
    {
        PyErr_SetString(PyExc_ValueError, generr.what());
        return NULL;
    }
    return nedToPy(type);
}
