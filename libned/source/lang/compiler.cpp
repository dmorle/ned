#include <ned/errors.h>
#include <ned/lang/compiler.h>

#include <map>
#include <unordered_map>
#include <filesystem>
#include <variant>

namespace fs = std::filesystem;

namespace nn
{
    namespace lang
    {
        // Resolves imports of an AstModule to build a single code module without external dependancies
        class CodeModule
        {
            enum NodeType
            {
                NODE = 0,
                STRUCT,
                FUNC,
                DEF
            };

            struct Node
            {
                using AttrType = std::variant<Node, AstStruct, AstFn, AstBlock>;
                std::map<std::string, std::vector<AttrType>> attrs;
                std::map<std::string, std::vector<AstBlock>> intrs;
                std::map<std::string, std::vector<AstInit>>  inits;
            };

            Node root;

            template<typename T>
            static bool merge_node(Node& dst, T& src)
            {
                for (auto& ns : src.namespaces)
                {
                    Node nd;
                    if (merge_node(nd, ns))
                        return true;
                    dst.attrs[ns.name].push_back(std::move(nd));
                }
                for (auto& agg : src.structs)
                    dst.attrs[agg.signature.name].push_back(std::move(agg));
                for (auto& fn : src.funcs)
                    dst.attrs[fn.signature.name].push_back(std::move(fn));
                for (auto& def : src.defs)
                    dst.attrs[def.signature.name].push_back(std::move(def));
                for (auto& intr : src.intrs)
                    dst.intrs[intr.signature.name].push_back(std::move(intr));
                for (auto& init : src.inits)
                    dst.inits[init.name].push_back(std::move(init));
            }

        public:
            bool merge_ast(AstModule& ast)
            {
                return merge_node(root, ast);
            }
        };

        bool create_module(CodeModule& mod, const AstModule& ast, const std::vector<std::string>& imp_dirs, std::vector<std::string> visited)
        {
            auto build_fname = [](const std::vector<std::string>& imp) -> std::string
            {
                std::stringstream ss;
                for (size_t i = 0; i < imp.size() - 1; i++)
                    ss << imp[i] << "/";
                ss << imp[imp.size() - 1] << ".nn";
                return ss.str();
            };

            auto merge_file = [&mod, imp_dirs, &visited](const std::string& fname) -> bool
            {
                TokenArray tarr;
                AstModule ast;
                return
                    lex_file(fname.c_str(), tarr) ||
                    parse_module(tarr, ast) ||
                    create_module(mod, ast, imp_dirs, visited) ||
                    mod.merge_ast(ast);
            };

            std::string curr_dir = fs::path(ast.fname).parent_path().string();
            
            for (const auto& imp : ast.imports)
            {
                // checking if the import is in the current directory
                std::string fname = build_fname(imp.imp);
                std::string curr_fname = curr_dir + fname;
                if (fs::is_regular_file(curr_fname))
                {
                    if (merge_file(curr_fname))
                        return true;
                    visited.push_back(curr_fname);
                    continue;
                }

                // look through the import directories to find it
                bool found = false;
                for (const auto& imp_dir : imp_dirs)
                {
                    std::string imp_fname = imp_dir + fname;
                    if (fs::is_regular_file(imp_fname))
                    {
                        if (merge_file(imp_fname))
                            return true;
                        visited.push_back(imp_fname);
                        found = true;
                        break;
                    }
                }
                if (found)
                    continue;

                std::stringstream ss;
                for (size_t i = 0; i < imp.imp.size() - 1; i++)
                    ss << imp.imp[i] << ".";
                ss << imp.imp[imp.imp.size() - 1];
                return error::compiler(imp.node_info, "Unresolved import '%'", ss.str());
            }
        }

        bool codegen(ByteCodeModule& bc, const AstModule& ast, const std::vector<std::string>& imp_dirs)
        {
            // Resolving imports to build a CodeModule object
            CodeModule mod;
            std::vector<std::string> visited = { ast.fname };
            if (create_module(mod, ast, imp_dirs, visited))
                return true;
        }
    }
}
