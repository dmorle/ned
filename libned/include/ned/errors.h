#ifndef NED_ERRORS_H
#define NED_ERRORS_H

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4267)
#if defined(_WIN64) && (!defined(__cplusplus) || __cplusplus == 199711L)
#define __cxx_ver _MSVC_LANG
#endif

#include <string>

namespace std
{
    std::string to_string(std::string val);
    std::string to_string(const char* val);
    std::string to_string(bool val);
}

namespace nn
{
    namespace lang
    {
        class Token;
        class ByteCodeDebugInfo;
        struct AstNodeInfo;
    }

    namespace error
    {
        template<typename... Args> struct _Format;

        template<typename T, typename... Tail>
        struct _Format<T, Tail...>
        {
            static std::string format(const std::string& fmt, T val, Tail... tail)
            {
                size_t pos = fmt.find('%');
                if (pos == std::string::npos)
                    return fmt;
                // Inefficient, but it should be used very rarely
                return _Format<Tail...>::format(fmt.substr(0, pos) + std::to_string(val) + fmt.substr(pos + 1), tail...);
            }
        };

        template<> struct _Format<> { static std::string format(const std::string& fmt) { return fmt; } };

        template<typename... Args>
        std::string format(const std::string& fmt, Args... args) { return _Format<Args...>::format(fmt, args...); }
        template<typename... Args>
        std::string format(const char* fmt, Args... args) { return format<Args...>(std::string(fmt), args...); }

        bool no_memory();
        void pop_last();

        template<typename... Args> bool general(const std::string& fmt, Args... args);
        template<> bool general<>(const std::string& errmsg);
        template<typename... Args>
        bool general<Args...>(const std::string& fmt, Args... args)
        {
            return error::general<>(format(fmt, args...));
        }

        template<typename... Args> bool syntax(const std::string& fname, size_t line_num, size_t col_num, const std::string& fmt, Args... args);
        template<> bool syntax<>(const std::string& fname, size_t line_num, size_t col_num, const std::string& errmsg);
        template<typename... Args>
        bool syntax<Args...>(const std::string& fname, size_t line_num, size_t col_num, const std::string& fmt, Args... args)
        {
            return error::syntax<>(fname, line_num, col_num, format(fmt, args...));
        }

        template<typename... Args> bool syntax(const lang::Token* tk, const std::string& fmt, Args... args);
        template<> bool syntax<>(const lang::Token* tk, const std::string& errmsg);
        template<typename... Args>
        bool syntax<Args...>(const lang::Token* tk, const std::string& fmt, Args... args)
        {
            return error::syntax<>(tk, format(fmt, args...));
        }

        template<typename... Args> bool syntax(const lang::Token& tk, const std::string& fmt, Args... args);
        template<> bool syntax<>(const lang::Token& tk, const std::string& errmsg);
        template<typename... Args>
        bool syntax<Args...>(const lang::Token& tk, const std::string& fmt, Args... args)
        {
            return error::syntax<>(tk, format(fmt, args...));
        }

        void bind_runtime_context(const lang::ByteCodeDebugInfo& debug_info, const size_t& pc);

        template<typename... Args> bool compiler(const lang::AstNodeInfo& info, const std::string& fmt, Args... args);
        template<> bool compiler<>(const lang::AstNodeInfo& info, const std::string& errmsg);
        template<typename... Args>
        bool compiler<Args...>(const lang::AstNodeInfo& info, const std::string& fmt, Args... args)
        {
            return error::compiler<>(info, format(fmt, args...));
        }

        template<typename... Args> bool runtime(const std::string& fmt, Args... args);
        template<> bool runtime<>(const std::string& errmsg);
        template<typename... Args>
        bool runtime<Args...>(const std::string& fmt, Args... args)
        {
            return error::runtime<>(format(fmt, args...));
        }

        template<typename... Args> bool graph(const std::string& fmt, Args... args);
        template<> bool graph<>(const std::string& errmsg);
        template<typename... Args>
        bool graph<Args...>(const std::string& fmt, Args... args)
        {
            return error::graph<>(format(fmt, args...));
        }

        void print();
    }
}

#endif
