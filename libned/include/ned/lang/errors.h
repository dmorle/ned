#ifndef NED_ERRORS_H
#define NED_ERRORS_H

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4267)
#if defined(_WIN64) && (!defined(__cplusplus) || __cplusplus == 199711L)
#define __cxx_ver _MSVC_LANG
#endif

#include <string>
#include <vector>
#if __cxx_ver >= 202002L
#include <format>
#else
namespace std {
    template<class... Args>
    string format(const string& fmt, const Args&...) { return fmt; }
}
#endif

namespace nn
{
    namespace lang
    {
        class Token;

        class Error
        {
            std::string fname;
            size_t line_num;
            size_t col_num;
            std::string errmsg;

        public:
            template<typename... Args> Error(const std::string& fname, size_t line_num, size_t col_num, const std::string& fmt, const Args&... args);
            template<typename... Args> Error(const Token* ptk, const std::string& fmt, const Args&... args);
            template<typename... Args> Error(const Token& tk, const std::string& fmt, const Args&... args);
        };

        template<class T>
        class ErrorList
        {
            std::vector<T> errs;

        public:
            template<typename... Args>
            bool add(const Args&... args) { errs.push_back(T{ args... }); return true; }
        };

        using Errors = ErrorList<Error>;
    }
}

#define _CRT_SECURE_NO_WARNINGS
#include <ned/lang/lexer.h>
// Goddamnit c++, really?  Never saw templated members with circular deps?

namespace nn
{
    namespace lang
    {
        template<typename... Args>
        Error::Error(const std::string& fname, size_t line_num, size_t col_num, const std::string& fmt, const Args&... args) :
            fname(fname), line_num(line_num), col_num(col_num), errmsg(std::format(fmt, args...)) {}

        template<typename... Args>
        Error::Error(const Token* ptk, const std::string& fmt, const Args&... args) :
            fname(ptk->fname), line_num(ptk->line_num), col_num(ptk->col_num), errmsg(std::format(fmt, args...)) {}

        template<typename... Args>
        Error::Error(const Token& tk, const std::string& fmt, const Args&... args) :
            fname(tk.fname), line_num(tk.line_num), col_num(tk.col_num), errmsg(std::format(fmt, args...)) {}
    }
}

#endif
