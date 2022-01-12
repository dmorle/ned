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
#include <iostream>

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

            void print()
            {
                std::cout
                    << errmsg
                    << "\nline: " << line_num
                    << " column: " << col_num
                    << " file: " << fname;
            }
        };

        template<typename T>
        class ErrorList
        {
            std::vector<T> errs;

        public:
            template<typename... Args>
            bool add(const Args&... args) { errs.push_back(T{ args... }); return true; }

            void print()
            {
                for (auto& e : errs)
                {
                    e.print();
                    std::cout << "\n\n";
                }
                std::cout << std::endl;
            }
        };

        using Errors = ErrorList<Error>;

        class ByteCodeDebugInfo
        {
        public:
            struct Record
            {
                size_t addr;
                std::string fname;
                size_t line_num, col_num;
            };

        private:
            std::vector<Record> instruction_records;
            friend class ByteCodeBody;

        public:
            template<typename... Args>
            bool add_error_at(Errors& errs, size_t pc, Args... args) const
            {
                size_t min = 0;
                size_t max = instruction_records.size();
                while (true)
                {
                    size_t idx = (max + min) / 2;
                    size_t idx_pc = instruction_records[idx].addr;
                    if (pc < idx_pc)
                        max = idx;
                    else if (idx_pc < pc)
                        min = idx;
                    else
                        return errs.add(
                            instruction_records[idx].fname,
                            instruction_records[idx].line_num,
                            instruction_records[idx].col_num,
                            args...
                        )
                }
            }
        };

        class CallStack;
        class ProgramHeap;
        struct ByteCode;
        class RuntimeErrors
        {
            Errors& errs;
            ByteCodeDebugInfo* debug_info;
            size_t* pc;

            friend bool exec(Errors& errs, CallStack& stack, ProgramHeap& heap, ByteCode& byte_code, std::string entry_point, core::Graph& graph);

        public:
            RuntimeErrors(Errors& errs) : errs(errs), debug_info(nullptr), pc(nullptr) {}
            RuntimeErrors(Errors& errs, ByteCodeDebugInfo& debug_info, size_t& pc) : errs(errs), debug_info(&debug_info), pc(&pc) {}

            template<typename... Args>
            bool add(Args... args)
            {
                if (debug_info && pc)
                    return debug_info->add_error_at(errs, *pc, args...);
                return errs.add("", 0ULL, 0ULL, args...);
            }
        };
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
