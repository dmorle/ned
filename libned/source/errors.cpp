#include <ned/errors.h>
#include <ned/lang/lexer.h>
#include <ned/lang/bytecode.h>
#include <ned/lang/compiler.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace std
{
    std::string to_string(std::string val) { return val; }
    std::string to_string(const char* val) { return val; }
    std::string to_string(bool val) { return val ? "true" : "false"; }
    std::string to_string(::nn::lang::TypeRef type) { return type->to_string(); }
}

namespace nn
{
    namespace error
    {
        std::vector<std::string> errors;

        bool no_memory()
        {
            errors.push_back("Error: Out of memory");
            return true;
        }

        void pop_last()
        {
            errors.pop_back();
        }

        template<>
        bool general<>(const std::string& errmsg)
        {
            std::stringstream ss;
            ss << "Error: " << errmsg;
            errors.push_back(ss.str());
            return true;
        }

        template<>
        bool syntax<>(const std::string& fname, size_t line_num, size_t col_num, const std::string& errmsg)
        {
            std::stringstream ss;
            ss  << "Syntax Error: " << errmsg
                << "\nline: " << line_num
                << " column: " << col_num
                << " file: " << fname;
            errors.push_back(ss.str());
            return true;
        }

        template<>
        bool syntax<>(const lang::Token* tk, const std::string& errmsg)
        {
            std::stringstream ss;
            ss  << "Syntax Error: " << errmsg
                << "\nline: " << tk->line_num
                << " column: " << tk->col_num
                << " file: " << tk->fname;
            errors.push_back(ss.str());
            return true;
        }

        template<>
        bool syntax<>(const lang::Token& tk, const std::string& errmsg)
        {
            std::stringstream ss;
            ss  << "Syntax Error: " << errmsg
                << "\nline: " << tk.line_num
                << " column: " << tk.col_num
                << " file: " << tk.fname;
            errors.push_back(ss.str());
            return true;
        }

        const lang::ByteCodeDebugInfo* pdebug_info = nullptr;
        const size_t* ppc = nullptr;
        void bind_runtime_context(const lang::ByteCodeDebugInfo& debug_info, const size_t& pc)
        {
            pdebug_info = &debug_info;
            ppc = &pc;
        }

        template<>
        bool compiler<>(const lang::AstNodeInfo& type, const std::string& errmsg)
        {
            std::stringstream ss;
            ss << "Compilation Error: " << errmsg
                << "\nline: " << type.line_start
                << " column: " << type.col_start
                << " file: " << type.fname;
            errors.push_back(ss.str());
            return true;
        }

        template<>
        bool runtime<>(const std::string& errmsg)
        {
            std::stringstream ss;
            ss << "Runtime Error: " << errmsg;
            if (pdebug_info && ppc)
            {
                const auto& rec = pdebug_info->at(*ppc - 1);  // Compensating for the automatic pc++ that happens
                ss  << "\nline: " << rec.line_num
                    << " column: " << rec.col_num
                    << " file: " << rec.fname;
            }
            errors.push_back(ss.str());
            return true;
        }

        template<>
        bool graph<>(const std::string& errmsg)
        {
            std::stringstream ss;
            ss << "Graph Error: " << errmsg;
            errors.push_back(ss.str());
            return true;
        }

        void print()
        {
            for (const auto& err : errors)
                std::cout << err << "\n\n";
            std::cout << std::endl;
        }
    }
}
