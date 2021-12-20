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
        class ParsingError
        {
            std::string errmsg;
            size_t line_num;
            size_t col_num;

        public:
            template<typename... Args>
            ParsingError(size_t line_num, size_t col_num, const std::string& fmt, const Args&... args)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            ParsingError(const Token* ptk, const std::string& fmt, const Args&... args)
            {
                this->line_num = ptk->line_num;
                this->col_num = ptk->col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            ParsingError(const Token& tk, const std::string& fmt, const Args&... args)
            {
                this->line_num = tk.line_num;
                this->col_num = tk.col_num;
                this->errmsg = std::format(fmt, args...);
            }
        };

        class ParsingErrors
        {
            std::vector<ParsingError> errs;

        public:
            ParsingErrors() {}

            template<typename... Args>
            bool add(const Args&... args) { errs.push_back(ParsingError(args...)); return true; }
        };
    }
}

#endif
