#include <pyned/pyned.h>

#include <pyned/lang/astobj.h>

#define CHECK_ARGNUM(n)                                                   \
    if (nargs != n)                                                       \
    {                                                                     \
        char buf[64];                                                     \
        sprintf(buf, "Expected "#n" argument(s), recieved %llu", nargs);  \
        PyErr_SetString(PyExc_ValueError, buf);                           \
        return NULL;                                                      \
    }

PyObject* parse_file(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    CHECK_ARGNUM(1);
    int fd = PyObject_AsFileDescriptor(args[0]);
    if (fd == -1)
        return NULL;

    AstObject* pAst = PyObject_New(AstObject, &AstObjectType);
    if (!pAst) return PyErr_NoMemory();

#ifdef WIN32
    fd = _dup(fd);
    if (fd == -1)
        return PyErr_SetFromErrno(PyExc_OSError);

    FILE* pf = _fdopen(fd, "rb");
#else
    FILE* pf = fdopen(dup(fd), "rb");
#endif

    impl::TokenArray tarr;
    impl::lex_file(pf, tarr);
    pAst->ast = impl::AstModule(tarr);

    return (PyObject*)pAst;
}
