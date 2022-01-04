#ifndef NED_ERRORS_H
#define NED_ERRORS_H

#include <ned/lang/lexer.h>
#if __cplusplus >= 202002L
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
        class Error
        {
            size_t line_num;
            size_t col_num;
            std::string errmsg;

        public:
            template<typename... Args>
            Error(size_t line_num, size_t col_num, const std::string& fmt, const Args&... args)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            Error(const Token* ptk, const std::string& fmt, const Args&... args)
            {
                this->line_num = ptk->line_num;
                this->col_num = ptk->col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            Error(const Token& tk, const std::string& fmt, const Args&... args)
            {
                this->line_num = tk.line_num;
                this->col_num = tk.col_num;
                this->errmsg = std::format(fmt, args...);
            }
        };

        template<class T>
        class ErrorList
        {
            std::vector<T> errs;

        public:
            template<typename... Args>
            bool add(const Args&... args) { errs.push_back(T(args...)); return true; }
        };

        using Errors = ErrorList<Error>;
    }
}

#endif
