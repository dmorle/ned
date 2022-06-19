#include <ned/errors.h>
#include <ned/lang/compiler.h>

#include <map>
#include <unordered_map>
#include <filesystem>
#include <variant>
#include <algorithm>
#include <functional>
#include <cassert>

namespace fs = std::filesystem;

#include <iostream>

namespace nn
{
    namespace lang
    {
        // Module context

        static CodeModule* mod = nullptr;
        static ByteCodeModule* bc = nullptr;
        static std::vector<std::string> cg_ns;  // the current namespace

        // Block context

        enum class BodyType
        {
            INVALID,
            STRUCT,
            DEF,
            FN,
            INTR
        };
        union BodySig
        {
            const AstCargSig* carg_sig;
            const AstBlockSig* block_sig;
            const AstFnSig* fn_sig;
        };
        static BodyType body_type = BodyType::INVALID;
        static BodySig body_sig = BodySig{ nullptr };
        static TypeManager* type_manager = nullptr;
        static ByteCodeBody* body = nullptr;

        // Set by codegen_line_while, codegen_line_for.  Read by codegen_line_break and codegen_line_continue
        // I don't want to pass it as an argument, because that would further pollute the signature of codegen_line_*
        struct LoopContext
        {
            Scope* scope = nullptr;
            std::string cont_label;
            std::string break_label;
        };
        static LoopContext* loop_ctx = nullptr;
        static std::string* ret_label = nullptr;

        // Function implementations

        template<typename T>
        bool CodeModule::merge_node(Node& dst, T& src)
        {
            // std::variant isn't working for the non-copyable Ast* types
            // TODO: custom implementation of CodeModule::Attr that doesn't depend on std::variant

            for (const AstNamespace& ns : src.namespaces)
            {
                CodeModule::Node nd;
                if (merge_node(nd, ns))
                    return true;
                dst.attrs[ns.name].push_back(std::move(nd));
            }
            for (const AstStruct& agg : src.structs)
                dst.attrs[agg.signature.name].push_back(&agg);
            for (const AstFn& fn : src.funcs)
                dst.attrs[fn.signature.name].push_back(&fn);
            for (const AstBlock& def : src.defs)
                dst.attrs[def.signature.name].push_back(&def);
            for (const AstBlock& intr : src.intrs)
                dst.intrs[intr.signature.name].push_back(&intr);
            for (const AstInit& init : src.inits)
                dst.inits[init.signature.name].push_back(&init);
            return false;
        }

        bool CodeModule::merge_ast(AstModule& ast)
        {
            return merge_node(root, ast);
        }

        bool CodeModule::create(CodeModule& mod, AstModule& ast, const std::vector<std::string>& imp_dirs, std::vector<std::string> visited)
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
                    create(mod, ast, imp_dirs, visited) ||
                    mod.merge_ast(ast);
            };

            mod.merge_ast(ast);
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

            return false;
        }

        bool CodeModule::LookupResult::empty()
        {
            return
                nodes.size() == 0 &&
                structs.size() == 0 &&
                fns.size() == 0 &&
                defs.size() == 0 &&
                intrs.size() == 0 &&
                inits.size() == 0;
        }

        bool CodeModule::lookup(const CodeModule::LookupCtx& ctx, const std::string& idn, CodeModule::LookupResult& result)
        {
            // dfs through the ast to find the attr closest to cg_ns
            if (ctx.it != ctx.end)
            {
                // try to get to the proper level namespace first.  If the fails, search through the current level
                for (auto& attr : ctx.nd.attrs[*ctx.it])
                    if (attr.index() == AttrType::NODE && !lookup({ std::get<Node>(attr), ctx.it + 1, ctx.end }, idn, result))
                        return false;
            }
            // Either there wasn't anything in the upper namespace, or we are currently at the most upper namespace
            if (ctx.nd.attrs.contains(idn))
            {
                for (const auto& attr : ctx.nd.attrs.at(idn))
                {
                    switch (attr.index())
                    {
                    case AttrType::NODE:
                        result.nodes.push_back(&std::get<Node>(attr));
                        break;
                    case AttrType::STRUCT:
                        result.structs.push_back(std::get<const AstStruct*>(attr));
                        break;
                    case AttrType::FUNC:
                        result.fns.push_back(std::get<const AstFn*>(attr));
                        break;
                    case AttrType::DEF:
                        result.defs.push_back(std::get<const AstBlock*>(attr));
                        break;
                    default:
                        assert(false);
                    }
                }
            }
            if (ctx.nd.intrs.contains(idn))
                result.intrs = ctx.nd.intrs.at(idn);
            if (ctx.nd.inits.contains(idn))
                result.inits = ctx.nd.inits.at(idn);
            if (result.empty())
                return true;  // Failed at the current namespace
            // The current namespace had something, so initialize result.ns and return false
            for (std::vector<std::string>::const_iterator it = ctx.it; it != ctx.end; it++)
                result.ns.push_back(*it);
            return false;
        }

        TypeRef::TypeRef(size_t ptr) : ptr(ptr) {}

        TypeRef::operator bool() const noexcept { return ptr; }
        TypeInfo* TypeRef::operator->() noexcept { return type_manager->get(ptr); }
        const TypeInfo* TypeRef::operator->() const noexcept { return type_manager->get(ptr); }
        TypeInfo& TypeRef::operator*() noexcept { return *type_manager->get(ptr); }
        const TypeInfo& TypeRef::operator*() const noexcept { return *type_manager->get(ptr); }

        const TypeRef TypeInfo::null = TypeRef();

        TypeInfo::TypeInfo() {}

        TypeInfo::TypeInfo(TypeInfo&& type) noexcept
        {
            do_move(std::move(type));
        }

        TypeInfo& TypeInfo::operator=(TypeInfo&& type) noexcept
        {
            if (&type == this)
                return *this;
            this->~TypeInfo();
            do_move(std::move(type));
            return *this;
        }

        TypeInfo::~TypeInfo()
        {
            switch (ty)
            {
            case TypeInfo::Type::INVALID:
            case TypeInfo::Type::TYPE:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                break;
            case TypeInfo::Type::ARRAY:
                type_array.~TypeInfoArray();
                break;
            case TypeInfo::Type::TUPLE:
                type_tuple.~TypeInfoTuple();
                break;
            case TypeInfo::Type::LOOKUP:
                type_lookup.~TypeInfoLookup();
                break;
            case TypeInfo::Type::CARGBIND:
                type_cargbind.~TypeInfoCargBind();
                break;
            case TypeInfo::Type::STRUCT:
                type_struct.~TypeInfoStruct();
                break;
            case TypeInfo::Type::INIT:
            case TypeInfo::Type::NODE:
            case TypeInfo::Type::BLOCK:
                break;
            case TypeInfo::Type::EDGE:
            case TypeInfo::Type::TENSOR:
                break;
            case TypeInfo::Type::DLTYPE:
                type_dltype.~TypeInfoDlType();
                break;
            case TypeInfo::Type::GENERIC:
                type_generic.~TypeInfoGeneric();
                break;
            case TypeInfo::Type::UNPACK:
                type_array.~TypeInfoArray();
                break;
            default:
                assert(false);
            }
        }

        void TypeInfo::do_move(TypeInfo&& type) noexcept
        {
            ty = type.ty;
            cat = type.cat;
            codegen = std::move(type.codegen);

            switch (ty)
            {
            case TypeInfo::Type::INVALID:
            case TypeInfo::Type::TYPE:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                break;
            case TypeInfo::Type::ARRAY:
                new (&type_array) decltype(type_array)(std::move(type.type_array));
                break;
            case TypeInfo::Type::TUPLE:
                new (&type_tuple) decltype(type_tuple)(std::move(type.type_tuple));
                break;
            case TypeInfo::Type::LOOKUP:
                new (&type_lookup) decltype(type_lookup)(std::move(type.type_lookup));
                break;
            case TypeInfo::Type::CARGBIND:
                new (&type_cargbind) decltype(type_cargbind)(std::move(type.type_cargbind));
                break;
            case TypeInfo::Type::STRUCT:
                new (&type_struct) decltype(type_struct)(std::move(type.type_struct));
                break;
            case TypeInfo::Type::INIT:
            case TypeInfo::Type::NODE:
            case TypeInfo::Type::BLOCK:
                break;
            case TypeInfo::Type::EDGE:
            case TypeInfo::Type::TENSOR:
                break;
            case TypeInfo::Type::DLTYPE:
                new (&type_dltype) decltype(type_dltype)(std::move(type.type_dltype));
                break;
            case TypeInfo::Type::GENERIC:
                new (&type_generic) decltype(type_generic)(std::move(type.type_generic));
                break;
            case TypeInfo::Type::UNPACK:
                new (&type_array) decltype(type_array)(std::move(type.type_array));
                break;
            default:
                assert(false);
            }
        }

        bool TypeInfo::check_pos() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_neg() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_add() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
            case TypeInfo::Type::ARRAY:
            case TypeInfo::Type::TUPLE:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_sub() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_mul() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_div() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_mod() const
        {
            return ty == TypeInfo::Type::INT;
        }

        bool TypeInfo::check_pow() const
        {
            return ty == TypeInfo::Type::INT || ty == TypeInfo::Type::FLOAT;
        }

        bool TypeInfo::check_eq() const
        {
            switch (ty)
            {
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            case TypeInfo::Type::ARRAY:
                return type_array.elem->check_eq();
            case TypeInfo::Type::TUPLE:
                for (TypeRef elem : type_tuple.elems)
                    if (!elem->check_eq())
                        return false;
                return true;
            }
            return false;
        }

        bool TypeInfo::check_ne() const
        {
            switch (ty)
            {
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            case TypeInfo::Type::ARRAY:
                return type_array.elem->check_ne();
            case TypeInfo::Type::TUPLE:
                for (TypeRef elem : type_tuple.elems)
                    if (!elem->check_ne())
                        return false;
                return true;
            }
            return false;
        }

        bool TypeInfo::check_ge() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_le() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_gt() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_lt() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_xstr() const
        {
            switch (ty)
            {
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            case TypeInfo::Type::ARRAY:
                return type_array.elem->check_xstr();
            case TypeInfo::Type::TUPLE:
                for (TypeRef elem : type_tuple.elems)
                    if (!elem->check_xstr())
                        return false;
                return true;
            }
            return false;
        }

        bool TypeInfo::check_xint() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_xflt() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_cpy() const
        {
            switch (ty)
            {
            case TypeInfo::Type::TYPE:
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
            case TypeInfo::Type::ARRAY:
            case TypeInfo::Type::TUPLE:
                return true;
            }
            return false;
        }

        bool TypeInfo::check_idx() const
        {
            return false;
        }

        bool TypeInfo::check_len() const
        {
            return ty == TypeInfo::Type::ARRAY || ty == TypeInfo::Type::TUPLE;
        }
 
        std::string TypeInfo::encode() const
        {
            switch (ty)
            {
            case TypeInfo::Type::INVALID:
                return "-";
            case TypeInfo::Type::TYPE:
                return std::string("y") + type_type.base->encode();
            case TypeInfo::Type::PLACEHOLDER:
                return "-";
            case TypeInfo::Type::FTY:
                return "w";
            case TypeInfo::Type::BOOL:
                return "b";
            case TypeInfo::Type::INT:
                return "i";
            case TypeInfo::Type::FLOAT:
                return "f";
            case TypeInfo::Type::STR:
                return "s";
            case TypeInfo::Type::ARRAY:
                return std::string("a") + type_array.elem->encode();
            case TypeInfo::Type::TUPLE:
            {
                std::stringstream ss;
                ss << "t";
                for (TypeRef elem : type_tuple.elems)
                    ss << elem->encode();
                ss << "0";
                return ss.str();
            }
            case TypeInfo::Type::LOOKUP:
                return "-";
            case TypeInfo::Type::CARGBIND:
                return "-";
            case TypeInfo::Type::STRUCT:
                return "0";  // TODO: implement structs
            case TypeInfo::Type::INIT:
                return "I";
            case TypeInfo::Type::NODE:
                return "N";
            case TypeInfo::Type::BLOCK:
                return "B";
            case TypeInfo::Type::EDGE:
                return "E";
            case TypeInfo::Type::TENSOR:
                return "T";
            case TypeInfo::Type::DLTYPE:
                return "-";
            case TypeInfo::Type::GENERIC:
                return "g";
            case TypeInfo::Type::UNPACK:
                return "-";
            }
            return "-";
        }

        std::string TypeInfo::to_string() const
        {
            // Leveraging the error formating code for this
            // these aren't actually errors
            using namespace nn::error;
            switch (ty)
            {
            case TypeInfo::Type::INVALID:
                return "INVALID";
            case TypeInfo::Type::TYPE:
                return format("type<%>", type_type.base->to_string());
            case TypeInfo::Type::PLACEHOLDER:
                return "placeholder";
            case TypeInfo::Type::FTY:
                return "fty";
            case TypeInfo::Type::BOOL:
                return "bool";
            case TypeInfo::Type::INT:
                return "int";
            case TypeInfo::Type::FLOAT:
                return "float";
            case TypeInfo::Type::STR:
                return "str";
            case TypeInfo::Type::ARRAY:
                return format("array<%>", type_array.elem->to_string());
            case TypeInfo::Type::TUPLE:
            {
                if (type_tuple.elems.size() == 0)
                    return "tuple<>";
                std::stringstream ss;
                ss << "tuple<" << type_tuple.elems.front()->to_string();
                for (size_t i = 1; i < type_tuple.elems.size(); i++)
                    ss << ", " << type_tuple.elems[i]->to_string();
                ss << ">";
                return ss.str();
            }
            case TypeInfo::Type::LOOKUP:
                return format("lookup(\"%\")", type_lookup.name);
            case TypeInfo::Type::CARGBIND:
                return "cargbind";
            case TypeInfo::Type::STRUCT:
                return "TODO: implement structs";
            case TypeInfo::Type::INIT:
                return "init";
            case TypeInfo::Type::NODE:
                return "node";
            case TypeInfo::Type::BLOCK:
                return "block";
            case TypeInfo::Type::EDGE:
                return "edge";
            case TypeInfo::Type::TENSOR:
                return "tensor";
            case TypeInfo::Type::DLTYPE:
                return "dltype";
            case TypeInfo::Type::GENERIC:
                return format("generic(\"%\")", type_generic.name);
            case TypeInfo::Type::UNPACK:
                return format("unpack<%>", type_array.elem->to_string());
            }
            return "ERROR TYPE";
        }

        bool TypeInfo::to_obj(const AstNodeInfo& node_info, TypeRef& type) const
        {
            TypeRef tmp;
            switch (ty)
            {
            case TypeInfo::Type::FTY:
                tmp = type_manager->create_fty(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        size_t val_addr, type_addr;
                        return
                            bc->add_obj_fty(val_addr, core::EdgeFty::F32) ||
                            bc->add_type_fty(type_addr) ||
                            body->add_instruction(instruction::New(node_info, val_addr)) ||
                            body->add_instruction(instruction::New(node_info, type_addr)) ||
                            body->add_instruction(instruction::Cpy(node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                        size_t addr;
                        return
                            bc->add_type_fty(addr) ||
                            body->add_instruction(instruction::New(node_info, addr)) ||
                            scope.push();
                    }, tmp);
                break;
            case TypeInfo::Type::BOOL:
                tmp = type_manager->create_bool(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        size_t val_addr, type_addr;
                        return
                            bc->add_obj_bool(val_addr, false) ||
                            bc->add_type_bool(type_addr) ||
                            body->add_instruction(instruction::New(node_info, val_addr)) ||
                            body->add_instruction(instruction::New(node_info, type_addr)) ||
                            body->add_instruction(instruction::Cpy(node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                        size_t addr;
                        return
                            bc->add_type_bool(addr) ||
                            body->add_instruction(instruction::New(node_info, addr)) ||
                            scope.push();
                    }, tmp);
                break;
            case TypeInfo::Type::INT:
                tmp = type_manager->create_int(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        size_t val_addr, type_addr;
                        return
                            bc->add_obj_int(val_addr, 0) ||
                            bc->add_type_int(type_addr) ||
                            body->add_instruction(instruction::New(node_info, val_addr)) ||
                            body->add_instruction(instruction::New(node_info, type_addr)) ||
                            body->add_instruction(instruction::Cpy(node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                        size_t addr;
                        return
                            bc->add_type_int(addr) ||
                            body->add_instruction(instruction::New(node_info, addr)) ||
                            scope.push();
                    }, tmp);
                break;
            case TypeInfo::Type::FLOAT:
                tmp = type_manager->create_float(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        size_t val_addr, type_addr;
                        return
                            bc->add_obj_float(val_addr, 0.0) ||
                            bc->add_type_float(type_addr) ||
                            body->add_instruction(instruction::New(node_info, val_addr)) ||
                            body->add_instruction(instruction::New(node_info, type_addr)) ||
                            body->add_instruction(instruction::Cpy(node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                        size_t addr;
                        return
                            bc->add_type_float(addr) ||
                            body->add_instruction(instruction::New(node_info, addr)) ||
                            scope.push();
                    }, tmp);
                break;
            case TypeInfo::Type::STR:
                tmp = type_manager->create_string(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        size_t val_addr, type_addr;
                        return
                            bc->add_obj_str(val_addr, "") ||
                            bc->add_type_str(type_addr) ||
                            body->add_instruction(instruction::New(node_info, val_addr)) ||
                            body->add_instruction(instruction::New(node_info, type_addr)) ||
                            body->add_instruction(instruction::Cpy(node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                        size_t addr;
                        return
                            bc->add_type_str(addr) ||
                            body->add_instruction(instruction::New(node_info, addr)) ||
                            scope.push();
                    }, tmp);
                break;
            case TypeInfo::Type::ARRAY:
            {
                tmp = type_manager->create_array(
                    TypeInfo::Category::DEFAULT,
                    [&node_info](Scope& scope) -> bool {
                        return
                            body->add_instruction(instruction::Agg(node_info, 0)) ||
                            scope.push();
                    }, type_array.elem);
                if (!tmp) return true;
                if (type_array.elem->ty == TypeInfo::Type::PLACEHOLDER)
                {
                    type = type_manager->create_type(
                        TypeInfo::Category::CONST,
                        [&node_info](Scope& scope) -> bool {
                            return
                                body->add_instruction(instruction::Nul(node_info)) ||
                                body->add_instruction(instruction::Arr(node_info)) ||
                                scope.push();
                        }, tmp);
                }
                else
                {
                    TypeRef elem;
                    if (type_array.elem->to_obj(node_info, elem))
                        return true;
                    type = type_manager->create_type(
                        TypeInfo::Category::CONST,
                        [&node_info, elem](Scope& scope) -> bool {
                            return
                                elem->codegen(scope) ||
                                body->add_instruction(instruction::Arr(node_info));
                        }, tmp);
                }
                break;
            }
            case TypeInfo::Type::TUPLE:
            {
                std::vector<TypeRef> elem_types;
                for (TypeRef elem : type_tuple.elems)
                {
                    TypeRef elem_type;
                    if (elem->to_obj(node_info, elem_type))
                        return true;
                    elem_types.push_back(elem_type);
                }
                tmp = type_manager->create_tuple(
                    TypeInfo::Category::DEFAULT,
                    [&node_info, elem_types](Scope& scope) -> bool {
                        for (TypeRef elem_type : elem_types)
                            if (elem_type->type_type.base->codegen(scope) ||
                                elem_type->codegen(scope) ||
                                body->add_instruction(instruction::Cpy(node_info)) ||
                                scope.pop()
                                ) return true;
                        if (body->add_instruction(instruction::Agg(node_info, elem_types.size())))
                            return true;
                        if (elem_types.size() == 0)
                            return scope.push();
                        return scope.pop(elem_types.size() - 1);
                    }, type_tuple.elems);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info, elem_types](Scope& scope) -> bool {
                        for (TypeRef elem_type : elem_types)
                            if (elem_type->codegen(scope))
                                return true;
                        if (body->add_instruction(instruction::Aty(node_info, elem_types.size())))
                            return true;
                        if (elem_types.size() == 0)
                            return scope.push();
                        return scope.pop(elem_types.size() - 1);
                    }, tmp);
                break;
            }
            default:
                return error::compiler(node_info, "Unable to transform the type % into a runtime object", to_string());
            }
            return !type;
        }

        bool operator==(const TypeRef& lhs, const TypeRef& rhs)
        {
            if (lhs->ty != rhs->ty)
                return false;
            switch (lhs->ty)
            {
            case TypeInfo::Type::TYPE:
                return lhs->type_type.base == rhs->type_type.base;
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                return true;
            case TypeInfo::Type::ARRAY:
                return lhs->type_array.elem == rhs->type_array.elem;
            case TypeInfo::Type::TUPLE:
                if (lhs->type_tuple.elems.size() != rhs->type_tuple.elems.size())
                    return false;
                for (size_t i = 0; i < lhs->type_tuple.elems.size(); i++)
                    if (lhs->type_tuple.elems[i] != rhs->type_tuple.elems[i])
                        return false;
                return true;
            case TypeInfo::Type::STRUCT:
                assert(false);
                // TODO: implement structs
                return false;
            case TypeInfo::Type::GENERIC:
                return lhs->type_generic.name == rhs->type_generic.name;
            }
            return false;
        }

        bool operator!=(const TypeRef& lhs, const TypeRef& rhs)
        {
            return !(lhs == rhs);
        }

        TypeInfo* TypeManager::get(size_t ptr) noexcept
        {
            return &buf[ptr];
        }

        const TypeInfo* TypeManager::get(size_t ptr) const noexcept
        {
            return &buf[ptr];
        }

        TypeRef TypeManager::next()
        {
            TypeRef ret = TypeRef(buf.size());
            buf.push_back(TypeInfo());
            return ret;
        }

        TypeManager::TypeManager()
        {
            buf.reserve(1024);
            buf.push_back(TypeInfo());  // null element
        }

        TypeRef TypeManager::duplicate(const TypeRef src)
        {
            return duplicate(src->cat, src->codegen, src);
        }

        TypeRef TypeManager::duplicate(TypeInfo::Category cat, CodegenCallback codegen, const TypeRef src)
        {
            TypeRef type = next();
            TypeRef tmp;
            if (!type) return TypeInfo::null;
            switch (src->ty)
            {
            case TypeInfo::Type::INVALID:
                assert(false);
                return TypeInfo::null;
            case TypeInfo::Type::TYPE:
                tmp = duplicate(src->type_type.base);
                if (!tmp) return TypeInfo::null;
                new (&type->type_type) TypeInfoType{ tmp };
                break;
            case TypeInfo::Type::PLACEHOLDER:
            case TypeInfo::Type::FTY:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                break;
            case TypeInfo::Type::ARRAY:
                tmp = duplicate(src->type_array.elem);
                if (!tmp) return TypeInfo::null;
                new (&type->type_array) TypeInfoArray{ tmp };
                break;
            case TypeInfo::Type::TUPLE:
                new (&type->type_tuple) TypeInfoTuple();
                type->ty = TypeInfo::Type::TUPLE;
                for (const TypeRef elem_type : src->type_tuple.elems)
                {
                    tmp = duplicate(elem_type);
                    if (!tmp) return TypeInfo::null;
                    type->type_tuple.elems.push_back(tmp);
                }
                break;
            case TypeInfo::Type::LOOKUP:
                new (&type->type_lookup) TypeInfoLookup(src->type_lookup);
                break;
            case TypeInfo::Type::CARGBIND:
                new (&type->type_cargbind) TypeInfoCargBind(src->type_cargbind);
                break;
            case TypeInfo::Type::STRUCT:
                // TODO: implement structs
                return TypeInfo::null;
            case TypeInfo::Type::INIT:
            case TypeInfo::Type::NODE:
            case TypeInfo::Type::BLOCK:
            case TypeInfo::Type::EDGE:
            case TypeInfo::Type::TENSOR:
                break;
            case TypeInfo::Type::DLTYPE:
                new (&type->type_dltype) TypeInfoDlType();
                type->ty = TypeInfo::Type::DLTYPE;
                tmp = duplicate(src->type_dltype.tensor);
                if (!tmp) return TypeInfo::null;
                type->type_dltype.tensor = tmp;
                tmp = duplicate(src->type_dltype.edge);
                if (!tmp) return TypeInfo::null;
                type->type_dltype.edge = tmp;
                tmp = duplicate(src->type_dltype.shape);
                if (!tmp) return TypeInfo::null;
                type->type_dltype.shape = tmp;
                tmp = duplicate(src->type_dltype.fp);
                if (!tmp) return TypeInfo::null;
                type->type_dltype.fp = tmp;
                break;
            case TypeInfo::Type::GENERIC:
                new (&type->type_generic) TypeInfoGeneric{ src->type_generic.name };
                break;
            case TypeInfo::Type::UNPACK:
                tmp = duplicate(src->type_array.elem);
                if (!tmp) return TypeInfo::null;
                new (&type->type_array) TypeInfoArray{ tmp };
                break;
            default:
                assert(false);
                return TypeInfo::null;
            }
            type->ty = src->ty;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_type(TypeInfo::Category cat, CodegenCallback codegen, TypeRef base)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_type) TypeInfoType{ base };
            type->ty = TypeInfo::Type::TYPE;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_placeholder()
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::PLACEHOLDER;
            type->cat = TypeInfo::Category::VIRTUAL;
            return type;
        }

        TypeRef TypeManager::create_generic(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::GENERIC;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_unpack(CodegenCallback codegen, TypeRef elem)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_array) TypeInfoArray{ elem };
            type->ty = TypeInfo::Type::UNPACK;
            type->cat = TypeInfo::Category::VIRTUAL;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_fty(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::FTY;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_bool(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::BOOL;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_int(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::INT;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_float(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::FLOAT;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_string(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::STR;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_array(TypeInfo::Category cat, CodegenCallback codegen, TypeRef elem)
        {
            TypeRef type = next();
            if (!type) return type;
            elem->cat = TypeInfo::Category::VIRTUAL;
            new (&type->type_array) TypeInfoArray{ elem };
            type->ty = TypeInfo::Type::ARRAY;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_tuple(TypeInfo::Category cat, CodegenCallback codegen, std::vector<TypeRef> elems)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_tuple) TypeInfoTuple{ elems };
            type->ty = TypeInfo::Type::TUPLE;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_lookup(std::string name, CodeModule::LookupResult lookup)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_lookup) TypeInfoLookup{ name, lookup };
            type->ty = TypeInfo::Type::LOOKUP;
            type->cat = TypeInfo::Category::VIRTUAL;
            type->codegen = nullptr;
            return type;
        }

        TypeRef TypeManager::create_cargbind(CodeModule::LookupResult lookup, const std::vector<AstExpr>& cargs)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_cargbind) TypeInfoCargBind{ lookup, cargs };
            type->ty = TypeInfo::Type::CARGBIND;
            type->cat = TypeInfo::Category::VIRTUAL;
            type->codegen = nullptr;
            return type;
        }

        TypeRef TypeManager::create_struct(TypeInfo::Category cat, CodegenCallback codegen)
        {
            error::general("Internal error: not implemented");
            return TypeInfo::null;
        }

        TypeRef TypeManager::create_init(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::INIT;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_node(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::NODE;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_block(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::BLOCK;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_edge(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::EDGE;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_tensor(TypeInfo::Category cat, CodegenCallback codegen)
        {
            TypeRef type = next();
            if (!type) return type;
            type->ty = TypeInfo::Type::TENSOR;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_dltype(TypeRef tensor, TypeRef edge, TypeRef shape, TypeRef fp)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_dltype) TypeInfoDlType{ tensor, edge, shape, fp };
            type->ty = TypeInfo::Type::DLTYPE;
            type->cat = TypeInfo::Category::VIRTUAL;
            type->codegen = nullptr;
            return type;
        }

        bool Scope::contains(const std::string& var_name) const
        {
            if (stack_vars.contains(var_name))
                return true;
            if (parent)
                return parent->contains(var_name);
            return false;
        }

        bool Scope::at(const std::string& var_name, Scope::StackVar& var) const
        {
            auto it = stack_vars.find(var_name);
            if (it != stack_vars.end())
            {
                var = it->second;
                return false;
            }
            if (parent)
                return parent->at(var_name, var);

            // not nessisary an error yet, it could be a variable declaration with automatic type deduction
            // so the responsibility is on the caller to generate any error messages
            return true;
        }

        bool Scope::add(const std::string& var_name, TypeRef type, const AstNodeInfo& node_info)
        {
            if (stack_vars.contains(var_name))  // I forgot what this was for, I might need to get rid of it
                return error::compiler(node_info, "Attempted to add variable '%' to the stack when it already exists", var_name);
            stack_vars[var_name] = { type, 0 };
            return false;
        }

        std::string Scope::generate_var_name()
        {
            while (contains(std::to_string(curr_var_name)))
                curr_var_name++;
            return std::to_string(curr_var_name);
        }

        bool Scope::empty() const
        {
            if (parent)
                return stack_vars.empty() && parent->empty();
            return stack_vars.empty();
        }

        bool Scope::push(size_t n)
        {
            for (auto& [name, var] : stack_vars)
                var.ptr += n;
            if (parent)
                parent->push(n);
            return false;
        }

        bool Scope::pop(size_t n)
        {
            std::vector<std::string> popped_vars{ stack_vars.size() };
            for (auto& [name, var] : stack_vars)
            {
                if (var.ptr < n)
                    popped_vars.push_back(name);
                else
                    var.ptr -= n;
            }
            if (parent)
                parent->pop(n);
            for (const auto& name : popped_vars)
                stack_vars.erase(name);
            return false;
        }

        const Scope* Scope::get_parent() const
        {
            return parent;
        }

        bool Scope::local_size(size_t& sz, const Scope* scope) const
        {
            sz += stack_vars.size();
            if (scope == parent)
                return false;
            if (!parent)
                return error::general("Internal error: invalid scope pointer");
            return parent->local_size(sz, scope);
        }

        bool Scope::list_local_vars(std::vector<StackVar>& vars, const Scope* scope)
        {
            size_t sz = 0;
            if (local_size(sz, scope))
                return true;
            vars.reserve(sz);

            // the result of this function will eventually be sorted in descending order
            // so I might as well try to help out with that a bit (parent vars will be at a higher depth than 'this')
            if (parent != scope)
            {
                assert(parent);  // this should already be checked by local_size()
                if (parent->list_local_vars(vars, scope))
                    return true;
            }
            for (auto& [name, var] : stack_vars)
                vars.push_back(var);
            assert(vars.size() == sz);
            return false;
        }

        ProcCall::ProcCall(const std::vector<std::string>& sig_ns) : sig_ns(sig_ns) {}

        ProcCall::~ProcCall()
        {
            for (size_t i = 0; i < val_buflen; i++)
                val_buf[i].~ValNode();
            for (size_t i = 0; i < type_buflen; i++)
                type_buf[i].~TypeNode();
        }

        ProcCall::ValNode::ValNode(ProcCall::ValNode&& node) noexcept
        {
            do_move(std::move(node));
        }

        ProcCall::ValNode& ProcCall::ValNode::operator=(ProcCall::ValNode&& node) noexcept
        {
            if (&node == this)
                return *this;
            this->~ValNode();
            do_move(std::move(node));
            return *this;
        }

        ProcCall::ValNode::~ValNode()
        {
            switch (ty)
            {
            case ValNode::Type::INVALID:
                break;
            case ValNode::Type::ARG_VAL:
                val_arg.~Arg();
                break;
            case ValNode::Type::CONST_VAL:
                break;
            case ValNode::Type::UNARY_POS:
            case ValNode::Type::UNARY_NEG:
            case ValNode::Type::UNARY_NOT:
            case ValNode::Type::UNARY_UNPACK:
                val_unary.~UnaryOp();
                break;
            case ValNode::Type::BINARY_ADD:
            case ValNode::Type::BINARY_SUB:
                val_binary.~BinaryOp();
                break;
            default:
                assert(false);
            }
        }

        void ProcCall::ValNode::do_move(ValNode&& node) noexcept
        {
            ty = node.ty;
            node.ty = ValNode::Type::INVALID;

            switch (ty)
            {
            case ValNode::Type::INVALID:
                break;
            case ValNode::Type::ARG_VAL:
                new (&val_arg) decltype(val_arg)(std::move(node.val_arg));
                break;
            case ValNode::Type::CONST_VAL:
                break;
            case ValNode::Type::UNARY_POS:
            case ValNode::Type::UNARY_NEG:
            case ValNode::Type::UNARY_NOT:
            case ValNode::Type::UNARY_UNPACK:
                new (&val_unary) decltype(val_unary)(std::move(node.val_unary));
                break;
            case ValNode::Type::BINARY_ADD:
            case ValNode::Type::BINARY_SUB:
                new (&val_binary) decltype(val_binary)(std::move(node.val_binary));
                break;
            default:
                assert(false);
            }
        }

        bool ProcCall::ValNode::get_type(const AstNodeInfo& node_info, std::vector<TypeRef>& rets) const
        {
            switch (ty)
            {
            case Type::INVALID:
                return error::compiler(node_info, "Internal error: invalid enum value");
            case Type::ARG_VAL:
                rets.push_back(val_arg.type->as_type(node_info));
                return !rets.back();
            case Type::CONST_VAL:
                if (val->ty == TypeInfo::Type::UNPACK)
                    rets.push_back(val->type_array.elem);
                else
                    rets.push_back(val);
                return false;
            case Type::UNARY_POS:
                if (val_unary.inp->get_type(node_info, rets))
                    return true;
                if (rets.size() != 1)
                    return error::compiler(node_info, "A positive operator requires exactly one argument");
                if (!rets[0]->check_pos())
                    return error::compiler(node_info, "Unable to compute the positive of the given type");
                return false;
            case Type::UNARY_NEG:
                if (val_unary.inp->get_type(node_info, rets))
                    return true;
                if (rets.size() != 1)
                    return error::compiler(node_info, "A negative operator requires exactly one argument");
                if (!rets[0]->check_neg())
                    return error::compiler(node_info, "Unable to compute the negative of the given type");
                return false;
            case Type::UNARY_NOT:
                if (val_unary.inp->get_type(node_info, rets))
                    return true;
                if (rets.size() != 1)
                    return error::compiler(node_info, "A not operator requires exactly one argument");
                if (rets[0]->ty != TypeInfo::Type::BOOL)
                    return error::compiler(node_info, "Unable to compute the not of the given type");
                return false;
            case Type::UNARY_UNPACK:
                if (val_unary.inp->get_type(node_info, rets))
                    return true;
                if (rets.size() != 1)
                    return false;  // Unpacking multiple rets is a noop
                {
                    TypeRef tmp = rets[0];
                    rets.clear();
                    if (tmp->ty == TypeInfo::Type::ARRAY)
                    {
                        rets.push_back(tmp->type_array.elem);
                        return false;
                    }
                    if (tmp->ty == TypeInfo::Type::TUPLE)
                    {
                        rets = tmp->type_tuple.elems;
                        return false;
                    }
                    return error::compiler(node_info, "Unable to unpack the given type");
                }
            }
            return error::compiler(node_info, "Internal error: enum out of range");
        }

        ProcCall::TypeNode::TypeNode(ProcCall::TypeNode&& node) noexcept
        {
            do_move(std::move(node));
        }

        ProcCall::TypeNode& ProcCall::TypeNode::operator=(ProcCall::TypeNode&& node) noexcept
        {
            if (&node == this)
                return *this;
            this->~TypeNode();
            do_move(std::move(node));
            return *this;
        }

        ProcCall::TypeNode::~TypeNode()
        {
            switch (ty)
            {
            case TypeNode::Type::INVALID:
            case TypeNode::Type::INIT:
            case TypeNode::Type::FTY:
            case TypeNode::Type::BOOL:
            case TypeNode::Type::INT:
            case TypeNode::Type::FLOAT:
            case TypeNode::Type::STRING:
                break;
            case TypeNode::Type::GENERIC:
            case TypeNode::Type::UNPACK:
                type_val.~ValType();
                break;
            case TypeNode::Type::ARRAY:
                type_array.~ArrayType();
                break;
            case TypeNode::Type::TUPLE:
                type_tuple.~TupleType();
                break;
            case TypeNode::Type::DLTYPE:
                type_dl.~DlType();
                break;
            default:
                assert(false);
            }
        }

        void ProcCall::TypeNode::do_move(TypeNode&& node) noexcept
        {
            ty = node.ty;
            node.ty = TypeNode::Type::INVALID;

            switch (ty)
            {
            case TypeNode::Type::INVALID:
            case TypeNode::Type::INIT:
            case TypeNode::Type::FTY:
            case TypeNode::Type::BOOL:
            case TypeNode::Type::INT:
            case TypeNode::Type::FLOAT:
            case TypeNode::Type::STRING:
                break;
            case TypeNode::Type::GENERIC:
            case TypeNode::Type::UNPACK:
                new (&type_val) decltype(type_val)(std::move(node.type_val));
                break;
            case TypeNode::Type::ARRAY:
                new (&type_array) decltype(type_array)(std::move(node.type_array));
                break;
            case TypeNode::Type::TUPLE:
                new (&type_tuple) decltype(type_tuple)(std::move(node.type_tuple));
                break;
            case TypeNode::Type::DLTYPE:
                new (&type_dl) decltype(type_dl)(std::move(node.type_dl));
                break;
            default:
                assert(false);
            }
        }

        TypeRef ProcCall::TypeNode::as_type(const AstNodeInfo& node_info) const
        {
            switch (ty)
            {
            case TypeNode::Type::INVALID:
                error::compiler(node_info, "Internal error: enumeration found in the invalid state");
                return TypeRef();
            case TypeNode::Type::TYPE:
                return type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, TypeRef());
            case TypeNode::Type::INIT:
                return type_manager->create_init(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::FTY:
                return type_manager->create_fty(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::BOOL:
                return type_manager->create_bool(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::INT:
                return type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::FLOAT:
                return type_manager->create_float(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::STRING:
                return type_manager->create_string(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::GENERIC:
                return type_manager->create_generic(TypeInfo::Category::VIRTUAL, nullptr);
            case TypeNode::Type::UNPACK:
                error::compiler(node_info, "Internal error: not implemented");
                return TypeRef();
            case TypeNode::Type::ARRAY:
            {
                TypeRef tmp = type_array.carg->as_type(node_info);
                if (!tmp)
                    return TypeRef();
                return type_manager->create_array(TypeInfo::Category::VIRTUAL, nullptr, tmp);
            }
            case TypeNode::Type::TUPLE:
            {
                std::vector<TypeRef> cargs;
                for (auto& e : type_tuple.cargs)
                {
                    cargs.push_back(e->as_type(node_info));
                    if (!cargs.back())
                        return TypeRef();
                }
                return type_manager->create_tuple(TypeInfo::Category::VIRTUAL, nullptr, cargs);
            }
            case TypeNode::Type::DLTYPE:
                return type_manager->create_dltype(TypeInfo::null, TypeInfo::null, TypeInfo::null, TypeInfo::null);
            default:
                // this code should be unreachable
                error::compiler(node_info, "Internal error: enumeration found outside valid range");
                return TypeRef();
            }
        }

        bool ProcCall::create_arg(Scope& scope, const AstArgDecl& decl, ValNode*& node)
        {
            node = next_val();
            new (&node->val_arg) Arg();
            node->ty = ValNode::Type::ARG_VAL;
            node->node_info = &decl.node_info;
            node->val_arg.name = decl.var_name;
            if (!decl.is_packed)
            {
                // Non-packed arguments can have default values
                if (decl.default_expr)
                {
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *decl.default_expr, node->val_arg.default_type))
                        return true;
                }
                return create_type(scope, *decl.type_expr, node->val_arg.type);
            }
            // The argument is packed, wrap the type in an array and make sure theres no defaults
            if (decl.default_expr)
                return error::compiler(decl.node_info, "Packed arguments cannot have default values");
            node->val_arg.type = next_type();
            if (!node->val_arg.type)
                return true;
            new (&node->val_arg.type->type_array) ArrayType();
            node->val_arg.type->ty = TypeNode::Type::ARRAY;
            node->val_arg.type->node_info = &decl.node_info;
            return create_type(scope, *decl.type_expr, node->val_arg.type->type_array.carg);
        }

        bool ProcCall::create_type(Scope& scope, const AstExpr& expr, TypeNode*& node)
        {
            node = next_type();
            node->node_info = &expr.node_info;
            switch (expr.ty)
            {
            case ExprType::CARGS_CALL:
                // expr.expr_call.callee->ty needs to be a builtin type or structure
                if (expr.expr_call.callee->ty == ExprType::KW)
                    switch (expr.expr_call.callee->expr_kw)
                    {
                    case ExprKW::ARRAY:
                        if (expr.expr_call.args.size() != 1)
                            return error::compiler(expr.node_info, "Array types require exactly one carg");
                        new (&node->type_array) ArrayType();
                        node->ty = TypeNode::Type::ARRAY;
                        return create_type(scope, expr.expr_call.args[0], node->type_array.carg);
                    case ExprKW::TUPLE:
                        return error::compiler(expr.node_info, "Internal error: not implemented");
                    case ExprKW::F16:
                    case ExprKW::F32:
                    case ExprKW::F64:
                        break;
                    default:
                        return error::compiler(expr.node_info, "Invalid use of keyword '%'", to_string(expr.expr_kw));
                    }
                {
                    // Attempt a create_value on the callee and see what we end up with
                    ValNode* pnode;
                    if (create_value(scope, *expr.expr_call.callee, pnode))
                        return true;

                    std::vector<TypeRef> callee_types;
                    if (pnode->get_type(expr.expr_call.callee->node_info, callee_types))
                        return true;
                    if (callee_types.size() != 1)
                        return error::compiler(expr.expr_call.callee->node_info, "Cargs callee needs to be exactly one value");

                    if (callee_types.front()->ty == TypeInfo::Type::FTY)
                    {
                        // Its a tensor type, this places constraints on the cargs
                        new (&node->type_dl) DlType();
                        node->ty = TypeNode::Type::DLTYPE;
                        node->type_dl.fp = pnode;
                        bool has_unpack = false;
                        for (const AstExpr& arg : expr.expr_call.args)
                        {
                            ValNode* val_node;
                            if (create_value(scope, arg, val_node))
                                return true;
                            std::vector<TypeRef> val_types;
                            if (val_node->get_type(expr.node_info, val_types))
                                return true;
                            for (TypeRef val_type : val_types)
                                if (val_type->ty != TypeInfo::Type::INT)
                                    return error::compiler(expr.node_info, "Invalid type given as tensor carg, expected int");
                            // Checking for unpacked behaviour
                            if (val_node->ty == ValNode::Type::UNARY_UNPACK)
                            {
                                // Not nessisarilly an array unpack, need to check the type of the value it unpacks
                                std::vector<TypeRef> unpack_rets;
                                if (val_node->val_unary.inp->get_type(expr.node_info, unpack_rets))
                                    return true;
                                assert(unpack_rets.size() == 1);
                                if (unpack_rets[0]->ty == TypeInfo::Type::ARRAY)
                                {
                                    // Only a single unpacked array is allowed.  Otherwise, carg deduction will be ambiguous
                                    if (has_unpack)
                                        return error::compiler(expr.node_info, "Only a single unpacked array is allowed in a carg");
                                    has_unpack = true;
                                }
                            }
                            node->type_dl.shape.push_back(val_node);
                        }
                        return false;
                    }

                    // Its either a struct or an error.  Since structs aren't implemented yet, generate an error regardless
                    return error::compiler(expr.node_info, "Internal error: not implemented");
                }
            case ExprType::KW:
                switch (expr.expr_kw)
                {
                case ExprKW::TYPE:
                    node->ty = TypeNode::Type::TYPE;
                    return false;
                case ExprKW::INIT:
                    node->ty = TypeNode::Type::INIT;
                    return false;
                case ExprKW::FTY:
                    node->ty = TypeNode::Type::FTY;
                    return false;
                case ExprKW::BOOL:
                    node->ty = TypeNode::Type::BOOL;
                    return false;
                case ExprKW::INT:
                    node->ty = TypeNode::Type::INT;
                    return false;
                case ExprKW::FLOAT:
                    node->ty = TypeNode::Type::FLOAT;
                    return false;
                case ExprKW::STR:
                    node->ty = TypeNode::Type::STRING;
                    return false;
                default:
                    return error::compiler(expr.node_info, "Invalid use of keyword '%'", to_string(expr.expr_kw));
                }
            case ExprType::VAR:
                if (carg_nodes.contains(expr.expr_string))
                {
                    std::vector<TypeRef> rets;
                    ValNode* val_node = carg_nodes.at(expr.expr_string);
                    val_node->is_root = false;
                    if (val_node->get_type(expr.node_info, rets))
                        return true;
                    if (rets.size() != 1)
                        return error::compiler(expr.node_info, "To use a carg as a type, the carg must be exactly one value");
                    if (rets.front()->ty != TypeInfo::Type::TYPE)
                        return error::compiler(expr.node_info, "Only generic type cargs can be used as a type");
                    new (&node->type_val) ValType();
                    node->ty = TypeNode::Type::GENERIC;
                    node->type_val.val = val_node;
                    return false;
                }
                // TODO: Implement structs in siguratures
                return error::compiler(expr.node_info, "structs in signatures has not yet been implemented");
            default:
                // everything else is invalid
                return error::compiler(expr.node_info, "Invalid type expression in signature");
            }
        }

        bool ProcCall::create_value(Scope& scope, const AstExpr& expr, ValNode*& node)
        {
            if (expr.ty == ExprType::VAR && carg_nodes.contains(expr.expr_string))
            {
                // Its a reference to a carg
                node = carg_nodes.at(expr.expr_string);
                node->is_root = false;
                return false;
            }

            node = next_val();
            node->node_info = &expr.node_info;
            node->is_root = false;
            switch (expr.ty)
            {
            case ExprType::UNARY_POS:
                new (&node->val_unary) UnaryOp();
                node->ty = ValNode::Type::UNARY_POS;
                node->val_unary.inp = new ValNode{};
                return create_value(scope, *expr.expr_unary.expr, node->val_unary.inp);
            case ExprType::UNARY_NEG:
                new (&node->val_unary) UnaryOp();
                node->ty = ValNode::Type::UNARY_NEG;
                node->val_unary.inp = new ValNode{};
                return create_value(scope, *expr.expr_unary.expr, node->val_unary.inp);
            case ExprType::UNARY_NOT:
                new (&node->val_unary) UnaryOp();
                node->ty = ValNode::Type::UNARY_NOT;
                node->val_unary.inp = new ValNode{};
                return create_value(scope, *expr.expr_unary.expr, node->val_unary.inp);
            case ExprType::UNARY_UNPACK:
                new (&node->val_unary) UnaryOp();
                node->ty = ValNode::Type::UNARY_UNPACK;
                node->val_unary.inp = new ValNode{};
                return create_value(scope, *expr.expr_unary.expr, node->val_unary.inp);
            case ExprType::BINARY_ADD:
                new (&node->val_binary) BinaryOp();
                node->ty = ValNode::Type::BINARY_ADD;
                node->val_binary.lhs = new ValNode{};
                node->val_binary.rhs = new ValNode{};
                return
                    create_value(scope, *expr.expr_binary.left, node->val_binary.lhs) ||
                    create_value(scope, *expr.expr_binary.right, node->val_binary.rhs);
            case ExprType::BINARY_SUB:
                new (&node->val_binary) BinaryOp();
                node->ty = ValNode::Type::BINARY_SUB;
                node->val_binary.lhs = new ValNode{};
                node->val_binary.rhs = new ValNode{};
                return
                    create_value(scope, *expr.expr_binary.left, node->val_binary.lhs) ||
                    create_value(scope, *expr.expr_binary.right, node->val_binary.rhs);
            }

            // if all else fails, try codegening the expr and getting a constant value from it
            node->ty = ValNode::Type::CONST_VAL;
            return codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr, node->val);
        }

        bool ProcCall::codegen_root_arg(Scope& scope, ValNode& node)
        {
            assert(node.ty == ValNode::Type::ARG_VAL);
            if (node.val_arg.visited)  // Checking if the nodes was already generated and on the stack
                return false;

            if (!node.val)
                node.val = node.val_arg.default_type;
            if (!node.val)
            {
                // I never need to bubble up codegen calls since in ProcCall::codegen_root_arg
                // since its only called on root nodes.  And by definition, root nodes aren't
                // depended on by other nodes, making it impossible for a codegen call to give
                // any root node a value (it can be deduced from the remaining signature)
                return error::compiler(*node.node_info, "Unable to determine a value for arg % in signature", node.val_arg.name);
            }
            // At this point, the types should be guarenteed to match.  This is only for deducing args
            return codegen_type(scope, *node.val_arg.type, node.val);
        }

        bool ProcCall::codegen_value_arg(Scope& scope, ValNode& node, TypeRef& val)
        {
            return codegen_type(scope, *node.val_arg.type, val);
        }

        bool ProcCall::codegen_value_pos(Scope& scope, ValNode& node, TypeRef& val)
        {
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_value_neg(Scope& scope, ValNode& node, TypeRef& val)
        {
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_value_not(Scope& scope, ValNode& node, TypeRef& val)
        {
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_value_unpack(Scope& scope, ValNode& node, TypeRef& val)
        {
            return codegen_value(scope, *node.val_unary.inp, val);
        }

        bool ProcCall::codegen_value_add(Scope& scope, ValNode& node, TypeRef& val)
        {
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_value_sub(Scope& scope, ValNode& node, TypeRef& val)
        {
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_value(Scope& scope, ValNode& node, TypeRef& val)
        {
            // Any time a node already has a value, it can be directly compared against the deduced value at runtime
            if (node.val)
            {
                // Ensure at runtime that the provided value and the current value agree
                if (!node.val->check_eq())
                    return error::compiler(*node.node_info, "Unable to compare the given type");

                TypeRef type;
                if (node.val->to_obj(*node.node_info, type))
                    return true;
                size_t err_addr;
                if (bc->add_obj_str(err_addr, "Value mismatch found during signature deduction"))
                    return true;
                std::string end_label = label_prefix(*node.node_info) + "_arg_check_end";
                return
                    node.val->codegen(scope) ||
                    val->codegen(scope) ||
                    type->codegen(scope) ||
                    body->add_instruction(instruction::Eq(*node.node_info)) ||
                    body->add_instruction(instruction::Brt(*node.node_info, end_label)) ||
                    body->add_instruction(instruction::New(*node.node_info, err_addr)) ||
                    body->add_instruction(instruction::Err(*node.node_info)) ||
                    body->add_label(*node.node_info, end_label) ||
                    scope.pop(3);
            }
            // If the node doesn't already have a value, it automatically gets assigned one
            node.val = val;

            switch (node.ty)
            {
            case ValNode::Type::INVALID:
                return error::compiler(*node.node_info, "Internal error: attempted to codegen an uninitialized node");
            case ValNode::Type::ARG_VAL:
                return codegen_value_arg(scope, node, val);
            case ValNode::Type::CONST_VAL:
                return error::compiler(*node.node_info, "Internal error: Found an uninitialized constant value node");
            case ValNode::Type::UNARY_POS:
                return codegen_value_pos(scope, node, val);
            case ValNode::Type::UNARY_NEG:
                return codegen_value_neg(scope, node, val);
            case ValNode::Type::UNARY_NOT:
                return codegen_value_not(scope, node, val);
            case ValNode::Type::UNARY_UNPACK:
                return codegen_value_unpack(scope, node, val);
            case ValNode::Type::BINARY_ADD:
                return codegen_value_add(scope, node, val);
            case ValNode::Type::BINARY_SUB:
                return codegen_value_sub(scope, node, val);
            default:
                return error::compiler(*node.node_info, "Internal error: enum out of range");
            }
        }

        bool ProcCall::codegen_type_type(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::TYPE)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_init(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::INIT)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_fty(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::FTY)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_bool(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::BOOL)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_int(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::INT)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_float(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::FLOAT)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_string(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::STR)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return false;
        }

        bool ProcCall::codegen_type_generic(Scope& scope, TypeNode& node, TypeRef& type)
        {
            TypeRef val;
            if (type->to_obj(*node.node_info, val))
                return true;
            return codegen_value(scope, *node.type_val.val, val);
        }

        bool ProcCall::codegen_type_array(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::ARRAY)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            return codegen_type(scope, *node.type_array.carg, type->type_array.elem);
        }

        bool ProcCall::codegen_type_tuple(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != TypeInfo::Type::TUPLE)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            node.type_tuple.cargs;
            type->type_tuple;
            return error::compiler(*node.node_info, "Internal error: not implemented");
        }

        bool ProcCall::codegen_type_dltype(Scope& scope, TypeNode& node, TypeRef& type)
        {
            if (type->ty != dltype)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            TypeRef fptype = get_fp(*node.node_info, type);
            if (!fptype)
                return true;
            if (codegen_value(scope, *node.type_dl.fp, fptype))
                return true;
            int unpack_idx = -1;
            for (int i = 0; i < node.type_dl.shape.size(); i++)
                if (node.type_dl.shape[i]->ty == ValNode::Type::UNARY_UNPACK)
                {
                    if (unpack_idx != -1)
                        return error::compiler(*node.type_dl.shape[i]->node_info, "Found multiple unpacks in tensor's shape");
                    unpack_idx = i;
                }
            if (unpack_idx == -1)
            {
                // Basic case, the shape of the tensor is described explicitly
                std::string end_lbl = label_prefix(*node.node_info) + "_shape_end";
                size_t sz_addr, int_addr, err_addr;
                if (bc->add_obj_int(sz_addr, node.type_dl.shape.size()) ||
                    bc->add_type_int(int_addr) ||
                    bc->add_obj_str(err_addr, "Tensor rank mismatch found during signature deduction")
                    ) return true;

                TypeRef shape_type = get_shape(*node.node_info, type);
                if (!shape_type)
                    return true;
                if (shape_type->codegen(scope) ||
                    body->add_instruction(instruction::New(*node.node_info, int_addr)) ||
                    body->add_instruction(instruction::Arr(*node.node_info)) ||
                    body->add_instruction(instruction::Len(*node.node_info)) ||
                    body->add_instruction(instruction::New(*node.node_info, sz_addr)) ||
                    body->add_instruction(instruction::New(*node.node_info, int_addr)) ||
                    body->add_instruction(instruction::Eq(*node.node_info)) ||
                    body->add_instruction(instruction::Brt(*node.node_info, end_lbl)) ||
                    body->add_instruction(instruction::New(*node.node_info, err_addr)) ||
                    body->add_instruction(instruction::Err(*node.node_info)) ||
                    body->add_label(*node.node_info, end_lbl) ||
                    scope.pop()
                    ) return true;
                // Confirmed that at runtime the shape matches
                for (size_t i = 0; i < node.type_dl.shape.size(); i++)
                {
                    size_t addr;
                    if (bc->add_obj_int(addr, i))
                        return true;
                    TypeRef elem_ty = type_manager->create_int(
                        TypeInfo::Category::CONST,
                        [node_info{ node.node_info }, &type, addr, int_addr, shape_type](Scope& scope) {
                        return
                            shape_type->codegen(scope) ||
                            body->add_instruction(instruction::New(*node_info, addr)) ||
                            body->add_instruction(instruction::New(*node_info, int_addr)) ||
                            body->add_instruction(instruction::Arr(*node_info)) ||
                            body->add_instruction(instruction::Idx(*node_info));
                    });
                    if (!elem_ty || codegen_value(scope, *node.type_dl.shape[i], elem_ty))
                        return true;
                }
                return false;
            }

            // There was an unpack, so theres just a minimum size constraint placed on the tensor's rank,
            // and the nodes peel elements off both the front and back of the shape

            // Initializing static objects for later use
            std::string end_lbl = label_prefix(*node.node_info) + "_shape_end";
            size_t sz_addr, int_addr, err_addr;
            if (bc->add_obj_int(sz_addr, node.type_dl.shape.size() - 1) ||
                bc->add_type_int(int_addr) ||
                bc->add_obj_str(err_addr, "Tensor rank mismatch found during signature deduction")
                ) return true;
            // Retrieving the shape of the given dltype to compare and check against
            TypeRef shape_type = get_shape(*node.node_info, type);
            if (!shape_type)
                return true;
            // Check to make sure the rank is large enough
            if (shape_type->codegen(scope) ||
                body->add_instruction(instruction::New(*node_info, int_addr)) ||
                body->add_instruction(instruction::Arr(*node_info)) ||
                body->add_instruction(instruction::Len(*node.node_info)) ||
                body->add_instruction(instruction::New(*node.node_info, sz_addr)) ||
                body->add_instruction(instruction::New(*node.node_info, int_addr)) ||
                // If the rank of the tensor is less than the number of explicit parameters
                // It's impossible to find a mapping, so generate a runtime error instead
                body->add_instruction(instruction::Lt(*node.node_info)) ||
                body->add_instruction(instruction::Brf(*node.node_info, end_lbl)) ||
                body->add_instruction(instruction::New(*node.node_info, err_addr)) ||
                body->add_instruction(instruction::Err(*node.node_info)) ||
                body->add_label(*node.node_info, end_lbl) ||
                scope.pop()
                ) return true;

            // Building the explicit parameters prior to the packed argument
            for (size_t i = 0; i < unpack_idx; i++)
            {
                size_t idx_addr;
                if (bc->add_obj_int(idx_addr, i))
                    return true;
                TypeRef elem_ty = type_manager->create_int(
                    TypeInfo::Category::CONST,
                    [node_info{ node.node_info }, shape_type, int_addr, idx_addr](Scope& scope) -> bool {
                    return
                        shape_type->codegen(scope) ||
                        body->add_instruction(instruction::New(*node_info, idx_addr)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::Idx(*node_info));
                });
                if (!elem_ty || codegen_value(scope, *node.type_dl.shape[i], elem_ty))
                    return true;
            }
            // Building the explicit parameters after the packed argument
            for (size_t i = unpack_idx + 1; i < node.type_dl.shape.size(); i++)
            {
                size_t idx_addr;
                if (bc->add_obj_int(idx_addr, node.type_dl.shape.size() - i))
                    return true;
                TypeRef elem_ty = type_manager->create_int(
                    TypeInfo::Category::CONST,
                    [node_info{ node.node_info }, shape_type, int_addr, idx_addr](Scope& scope) -> bool {
                    return
                        shape_type->codegen(scope) ||
                        body->add_instruction(instruction::Dup(*node_info, 0)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::Len(*node_info)) ||
                        body->add_instruction(instruction::New(*node_info, idx_addr)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Sub(*node_info)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::Idx(*node_info));
                });
                if (!elem_ty || codegen_value(scope, *node.type_dl.shape[i], elem_ty))
                    return true;
            }

            // Building the packed argument
            size_t start_addr, end_addr, one_addr;
            if (bc->add_obj_int(start_addr, unpack_idx) ||
                bc->add_obj_int(end_addr, node.type_dl.shape.size() - unpack_idx - 1) ||
                bc->add_obj_int(one_addr, 1)
                ) return true;
            TypeRef tmp = type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
            if (!tmp)
                return true;
            TypeRef elem_ty = type_manager->create_array(
                TypeInfo::Category::CONST, [node_info{ node.node_info }, shape_type, int_addr, start_addr, end_addr, one_addr](Scope& scope) -> bool {
                    std::string start_label = label_prefix(*node_info) + "_packed_loop_start";
                    std::string end_label = label_prefix(*node_info) + "_packed_loop_end";
                    return
                        shape_type->codegen(scope) ||
                        body->add_instruction(instruction::Dup(*node_info, 0)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::Len(*node_info)) ||
                        body->add_instruction(instruction::New(*node_info, end_addr)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Sub(*node_info)) ||
                        body->add_instruction(instruction::New(*node_info, start_addr)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Cpy(*node_info)) ||
                        body->add_instruction(instruction::Agg(*node_info, 0)) ||
                        // stack is: shape, n, i, result
                        // if i == n: break
                        body->add_label(*node_info, start_label) ||
                        body->add_instruction(instruction::Dup(*node_info, 2)) ||
                        body->add_instruction(instruction::Dup(*node_info, 2)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Eq(*node_info)) ||
                        body->add_instruction(instruction::Brt(*node_info, end_label)) ||
                        // result += [shape[i]]
                        body->add_instruction(instruction::Dup(*node_info, 0)) ||
                        body->add_instruction(instruction::Dup(*node_info, 4)) ||
                        body->add_instruction(instruction::Dup(*node_info, 3)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::Idx(*node_info)) ||
                        body->add_instruction(instruction::Agg(*node_info, 1)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::Arr(*node_info)) ||
                        body->add_instruction(instruction::IAdd(*node_info)) ||
                        // i++
                        body->add_instruction(instruction::Dup(*node_info, 1)) ||
                        body->add_instruction(instruction::New(*node_info, one_addr)) ||
                        body->add_instruction(instruction::New(*node_info, int_addr)) ||
                        body->add_instruction(instruction::IAdd(*node_info)) ||
                        body->add_instruction(instruction::Jmp(*node_info, start_label)) ||
                        body->add_label(*node_info, end_label) ||
                        body->add_instruction(instruction::Pop(*node_info, 1)) ||
                        body->add_instruction(instruction::Pop(*node_info, 1)) ||
                        body->add_instruction(instruction::Pop(*node_info, 1));
                }, tmp);
            return !elem_ty || codegen_value(scope, *node.type_dl.shape[unpack_idx], elem_ty);
        }

        bool ProcCall::codegen_type(Scope& scope, TypeNode& node, TypeRef& type)
        {
            switch (node.ty)
            {
            case TypeNode::Type::TYPE:
                return codegen_type_type(scope, node, type);
            case TypeNode::Type::INIT:
                return codegen_type_init(scope, node, type);
            case TypeNode::Type::FTY:
                return codegen_type_fty(scope, node, type);
            case TypeNode::Type::BOOL:
                return codegen_type_bool(scope, node, type);
            case TypeNode::Type::INT:
                return codegen_type_int(scope, node, type);
            case TypeNode::Type::FLOAT:
                return codegen_type_float(scope, node, type);
            case TypeNode::Type::STRING:
                return codegen_type_string(scope, node, type);
            case TypeNode::Type::GENERIC:
                return codegen_type_generic(scope, node, type);
            case TypeNode::Type::ARRAY:
                return codegen_type_array(scope, node, type);
            case TypeNode::Type::TUPLE:
                return codegen_type_tuple(scope, node, type);
            case TypeNode::Type::DLTYPE:
                return codegen_type_dltype(scope, node, type);
            default:
                return error::compiler(*node.node_info, "Internal error: enum out of range");
            }
        }

        ProcCall::ValNode* ProcCall::next_val()
        {
            if (val_buflen == bufsz)
            {
                error::general("Exceeded the maximum number of value node allowed in the signature: 1024");
                return nullptr;
            }

            ValNode* ret = new (val_buf + val_buflen) ValNode();
            ret->ty = ValNode::Type::INVALID;
            val_buflen++;
            return ret;
        }

        ProcCall::TypeNode* ProcCall::next_type()
        {
            if (type_buflen == bufsz)
            {
                error::general("Exceeded the maximum number of type node allowed in the signature: 1024");
                return nullptr;
            }

            TypeNode* ret = new (type_buf + type_buflen) TypeNode();
            ret->ty = TypeNode::Type::INVALID;
            type_buflen++;
            return ret;
        }

        TensorCall::TensorCall(const std::vector<std::string>& sig_ns) :
            ProcCall(sig_ns)
        {
            dltype = TypeInfo::Type::TENSOR;
        }

        TypeRef TensorCall::get_fp(const AstNodeInfo& node_info, TypeRef type)
        {
            assert(type->ty == TypeInfo::Type::TENSOR);
            return type_manager->create_fty(
                TypeInfo::Category::DEFAULT,
                [&node_info, type](Scope& scope) -> bool {
                    return
                        type->codegen(scope) ||
                        body->add_instruction(instruction::Tfty(node_info));
                });
        }

        TypeRef TensorCall::get_shape(const AstNodeInfo& node_info, TypeRef type)
        {
            assert(type->ty == TypeInfo::Type::TENSOR);
            TypeRef tmp = type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
            if (!tmp)
                return tmp;
            return type_manager->create_array(
                TypeInfo::Category::DEFAULT,
                [&node_info, type](Scope& scope) -> bool {
                    return
                        type->codegen(scope) ||
                        body->add_instruction(instruction::Tshp(node_info));
                }, tmp);
        }

        EdgeCall::EdgeCall(const std::vector<std::string>& sig_ns) :
            ProcCall(sig_ns)
        {
            dltype = TypeInfo::Type::EDGE;
        }

        TypeRef EdgeCall::get_fp(const AstNodeInfo& node_info, TypeRef type)
        {
            assert(type->ty == TypeInfo::Type::EDGE);
            return type_manager->create_fty(
                TypeInfo::Category::DEFAULT,
                [&node_info, type](Scope& scope) -> bool {
                    return
                        type->codegen(scope) ||
                        body->add_instruction(instruction::Efty(node_info));
                });
        }

        TypeRef EdgeCall::get_shape(const AstNodeInfo& node_info, TypeRef type)
        {
            assert(type->ty == TypeInfo::Type::EDGE);
            TypeRef tmp = type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
            if (!tmp)
                return tmp;
            return type_manager->create_array(
                TypeInfo::Category::DEFAULT,
                [&node_info, type](Scope& scope) -> bool {
                    return
                        type->codegen(scope) ||
                        body->add_instruction(instruction::Eshp(node_info));
                }, tmp);
        }

        bool InitCall::init(const AstInit& sig)
        {
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;
            cg_ns = sig_ns;

            init_name = sig.signature.name;
            pinfo = &sig.node_info;

            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                // This should be a noop since this should've been checked during signature matching
                // But I'll keep it here anyway in case I do some sort of optimization in the future
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);
                // Constructing the ValNode for the argument
                ValNode* node;
                TypeRef type;
                if (create_arg(init_scope, arg_decl, node) ||
                    arg_type(type, init_scope, arg_decl) ||
                    init_scope.add(arg_decl.var_name, type, arg_decl.node_info) ||
                    init_scope.push()
                    ) return error::compiler(arg_decl.node_info, "Unable to to construct a carg node for parameter %", arg_decl.var_name);
                carg_nodes[arg_decl.var_name] = node;
                carg_stack.push_back(arg_decl.var_name);
            }

            cg_ns = old_ns;
            return false;
        }

        bool InitCall::apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs)
        {
            // Checking to make sure all the provided cargs are in the signature
            for (const auto& [name, type] : cargs)
                if (!carg_nodes.contains(name))
                    return error::compiler(node_info, "The provided carg '%' does not exist in the signature", name);

            // Putting the carg exprs into the nodes
            for (const auto& [name, expr] : cargs)
                carg_nodes[name]->val = expr;

            return false;
        }

        bool InitCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;

            // Creating the init object
            size_t addr;
            if (bc->add_obj_str(addr, init_name) ||
                body->add_instruction(instruction::New(*pinfo, addr)) ||
                body->add_instruction(instruction::Ini(*pinfo)) ||
                scope.push()
                ) return true;

            // Iterating through the cargs, configuring the init
            for (const auto& [name, node] : carg_nodes)
            {
                // Determining where to get the value of the arg from
                TypeRef val;
                if (node->val)
                    val = node->val;
                else if (node->val_arg.default_type)
                    val = node->val_arg.default_type;
                else
                    return error::compiler(*node->node_info, "Unable to deduce a value for carg '%'", name);
                
                TypeRef type;
                size_t name_addr;
                if (val->to_obj(*node->node_info, type) ||
                    bc->add_obj_str(name_addr, name) ||
                    val->codegen(scope) ||
                    type->codegen(scope) ||
                    body->add_instruction(instruction::New(*node->node_info, name_addr)) ||
                    body->add_instruction(instruction::InCfg(*node->node_info)) ||
                    scope.pop(2)
                    ) return error::compiler(*node->node_info, "Unable to compile the configuration for init carg '%'", name);
            }
            
            std::string var_name = scope.generate_var_name();
            TypeRef ret = type_manager->create_init(
                TypeInfo::Category::DEFAULT,
                [var_name, node_info{ this->node_info }](Scope& scope) {
                    Scope::StackVar var;
                    if (scope.at(var_name, var))
                        return error::compiler(*node_info, "Unable to resolve identifier %", var_name);
                    return
                        body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                        scope.push();
                });
            if (!ret || scope.add(var_name, ret, *node_info))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool IntrCall::init(const AstBlock& sig)
        {
            ast_intr = &sig;
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;
            cg_ns = sig_ns;

            // Building the nodes for the cargs
            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);

                ValNode* node;
                TypeRef type;
                if (create_arg(init_scope, arg_decl, node) ||
                    arg_type(type, init_scope, arg_decl) ||
                    init_scope.add(arg_decl.var_name, type, arg_decl.node_info) ||
                    init_scope.push()
                    ) return true;

                if (arg_decl.default_expr)
                {
                    if (arg_decl.is_packed)
                        return error::compiler(arg_decl.node_info, "Packed arguments cannot have default values");
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(init_scope, *arg_decl.default_expr, node->val_arg.default_type))
                        return true;
                }
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                carg_nodes[arg_decl.var_name] = node;
                carg_stack.push_back(arg_decl.var_name);
            }

            // Building the nodes for the vargs
            for (const auto& arg_decl : sig.signature.vargs)
            {
                // There shouldn't be any name collisions between all args
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' between cargs and vargs", arg_decl.var_name);
                if (varg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in vargs", arg_decl.var_name);

                ValNode* node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                if (arg_decl.default_expr)
                    return error::compiler(arg_decl.default_expr->node_info, "Default arguments are not allowed in vargs");
                if (node->val_arg.type->ty != TypeNode::Type::DLTYPE)
                    return error::compiler(*node->val_arg.type->node_info, "Only edge types are allowed as vargs in an intr");
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                varg_nodes[arg_decl.var_name] = node;
                varg_stack.push_back(arg_decl.var_name);
            }

            // Building the rets of the def
            for (const std::string& ret : sig.signature.rets)
            {
                if (std::find(ret_stack.begin(), ret_stack.end(), ret) != ret_stack.end())
                    return error::compiler(sig.node_info, "Naming conflict for return value '%'", ret);
                ret_stack.push_back(ret);
            }

            cg_ns = old_ns;  // Putting the old namespace back
            return false;
        }

        bool IntrCall::apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs)
        {
            // Checking to make sure all the provided cargs are in the signature
            for (const auto& [name, type] : cargs)
                if (!carg_nodes.contains(name))
                    return error::compiler(node_info, "The provided carg '%' does not exist in the signature", name);
            // Checking to make sure all the vargs are provided
            if (vargs.size() < varg_stack.size())
                return error::compiler(node_info, "Too few vargs provided in call");
            if (vargs.size() > varg_stack.size())
                return error::compiler(node_info, "Too many vargs provided in call");
            assert(vargs.size() == varg_stack.size());

            // Putting the carg exprs into the nodes
            for (const auto& [name, expr] : cargs)
                carg_nodes[name]->val = expr;

            // Putting the varg exprs into the nodes
            for (size_t i = 0; i < vargs.size(); i++)
                varg_nodes[varg_stack[i]]->val = vargs[i];
            return false;
        }

        bool IntrCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;
            for (auto& [name, node] : varg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;

            // codegening each of the cargs onto the stack
            for (const std::string& name : carg_stack)
            {
                const ValNode& node = *carg_nodes.at(name);
                TypeRef val;
                if (node.val)
                    val = node.val;
                else if (node.val_arg.default_type)
                    val = node.val_arg.default_type;
                else
                    return error::compiler(*node.node_info, "Unable to deduce a value for carg '%'", name);

                if (val->codegen(scope))
                    return true;
            }

            // codegening each of the vargs onto the stack
            for (const std::string& name : varg_stack)
            {
                const ValNode& node = *varg_nodes.at(name);
                if (!node.val)
                    return error::compiler(*node.node_info, "Missing required varg '%'", name);

                if (node.val->codegen(scope))
                    return true;
            }

            // calling the procedure
            std::string proc_name;
            size_t proc_addr;
            Scope::StackVar var;
            if (proc_name_intr(proc_name, sig_ns, *ast_intr) ||
                bc->add_static_ref(*node_info, proc_name, proc_addr) ||
                scope.at("~block", var) ||
                body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                body->add_instruction(instruction::New(*node_info, proc_addr)) ||
                body->add_instruction(instruction::Call(*node_info))
                ) return true;

            // cleaning the stack (+1 from ~block), and marking the return values to the stack
            for (size_t i = 0; i < carg_stack.size() + varg_stack.size() + 1; i++)
                if (body->add_instruction(instruction::Pop(*node_info, ret_stack.size())))
                    return true;
            if (scope.pop(carg_stack.size() + varg_stack.size()))
                return true;
            for (const std::string& ret_name : ret_stack)
            {
                std::string var_name = scope.generate_var_name();
                TypeRef ret = type_manager->create_edge(
                    TypeInfo::Category::DEFAULT,
                    [node_info{ node_info }, var_name](Scope& scope) -> bool {
                        Scope::StackVar var;
                        return
                            scope.at(var_name, var) ||
                            body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                            scope.push();
                    });
                if (!ret || scope.push() || scope.add(var_name, ret, *node_info))
                    return true;
                rets.push_back(ret);
            }
            return false;
        }
        
        bool DefCall::init(const AstBlock& sig)
        {
            ast_def = &sig;
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;
            cg_ns = sig_ns;

            // Building the nodes for the cargs
            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);

                ValNode* node;
                TypeRef type;
                if (create_arg(init_scope, arg_decl, node) ||
                    arg_type(type, init_scope, arg_decl) ||
                    init_scope.add(arg_decl.var_name, type, arg_decl.node_info) ||
                    init_scope.push()
                    ) return true;

                if (arg_decl.default_expr)
                {
                    if (arg_decl.is_packed)
                        return error::compiler(arg_decl.node_info, "Packed arguments cannot have default values");
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(init_scope, *arg_decl.default_expr, node->val_arg.default_type))
                        return true;
                }
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                carg_nodes[arg_decl.var_name] = node;
                carg_stack.push_back(arg_decl.var_name);
            }

            // Building the nodes for the vargs
            for (const auto& arg_decl : sig.signature.vargs)
            {
                // There shouldn't be any name collisions between all args
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' between cargs and vargs", arg_decl.var_name);
                if (varg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in vargs", arg_decl.var_name);

                ValNode* node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                if (arg_decl.default_expr)
                    return error::compiler(arg_decl.default_expr->node_info, "Default arguments are not allowed in vargs");
                if (node->val_arg.type->ty != TypeNode::Type::DLTYPE)
                    return error::compiler(*node->val_arg.type->node_info, "Only tensor types are allowed as vargs in a def");
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                varg_nodes[arg_decl.var_name] = node;
                varg_stack.push_back(arg_decl.var_name);
            }

            // Building the rets of the def
            for (const std::string& ret : sig.signature.rets)
            {
                if (std::find(ret_stack.begin(), ret_stack.end(), ret) != ret_stack.end())
                    return error::compiler(sig.node_info, "Naming conflict for return value '%'", ret);
                ret_stack.push_back(ret);
            }

            cg_ns = old_ns;  // Putting the old namespace back
            return false;
        }

        bool DefCall::apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs)
        {
            // Checking to make sure all the provided cargs are in the signature
            for (const auto& [name, type] : cargs)
                if (!carg_nodes.contains(name))
                    return error::compiler(node_info, "The provided carg '%' does not exist in the signature", name);
            // Checking to make sure all the vargs are provided
            if (vargs.size() < varg_stack.size())
                return error::compiler(node_info, "Too few vargs provided in call");
            if (vargs.size() > varg_stack.size())
                return error::compiler(node_info, "Too many vargs provided in call");
            assert(vargs.size() == varg_stack.size());

            // Putting the carg exprs into the nodes
            for (const auto& [name, expr] : cargs)
                carg_nodes[name]->val = expr;

            // Putting the varg exprs into the nodes
            for (size_t i = 0; i < vargs.size(); i++)
                varg_nodes[varg_stack[i]]->val = vargs[i];
            return false;
        }

        bool DefCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {

            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;
            for (auto& [name, node] : varg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;
            
            // codegening each of the cargs onto the stack
            for (const std::string& name : carg_stack)
            {
                const ValNode& node = *carg_nodes.at(name);
                TypeRef val;
                if (node.val)
                    val = node.val;
                else if (node.val_arg.default_type)
                    val = node.val_arg.default_type;
                else
                    return error::compiler(*node.node_info, "Unable to deduce a value for carg '%'", name);
                
                if (val->codegen(scope))
                    return true;
            }

            // codegening each of the vargs onto the stack
            for (const std::string& name : varg_stack)
            {
                const ValNode& node = *varg_nodes.at(name);
                if (!node.val)
                    return error::compiler(*node.node_info, "Missing required varg '%'", name);
                
                if (node.val->codegen(scope))
                    return true;
            }

            // calling the procedure
            std::string proc_name;
            size_t proc_addr;
            Scope::StackVar var;
            if (proc_name_def(proc_name, sig_ns, *ast_def) ||
                bc->add_static_ref(*node_info, proc_name, proc_addr) ||
                scope.at("~block", var) ||
                body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                body->add_instruction(instruction::New(*node_info, proc_addr)) ||
                body->add_instruction(instruction::Call(*node_info))
                ) return true;

            // cleaning the stack (+1 from ~block), and marking the return values to the stack
            for (size_t i = 0; i < carg_stack.size() + varg_stack.size() + 1; i++)
                if (body->add_instruction(instruction::Pop(*node_info, ret_stack.size())))
                    return true;
            if (scope.pop(carg_stack.size() + varg_stack.size()))
                return true;
            for (const std::string& ret_name : ret_stack)
            {
                std::string var_name = scope.generate_var_name();
                TypeRef ret = type_manager->create_tensor(
                    TypeInfo::Category::DEFAULT,
                    [node_info{ node_info }, var_name](Scope& scope) -> bool {
                        Scope::StackVar var;
                        if (scope.at(var_name, var))
                            return error::compiler(*node_info, "Unable to resolve identifier %", var_name);
                        return
                            body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                            scope.push();
                    });
                if (!ret || scope.push() || scope.add(var_name, ret, *node_info))
                    return true;
                rets.push_back(ret);
            }
            return false;
        }

        bool DefCall::codegen_entrypoint(Scope& scope, const std::map<std::string, TypeRef>& cargs)
        {
            // Checking to make sure all the provided cargs are in the signature
            for (const auto& [name, carg] : cargs)
                if (!carg_nodes.contains(name))
                    return error::general("The provided carg '%' does not exist in the signature", name);

            // Putting the carg data into the nodes
            for (const auto& [name, carg] : cargs)
                carg_nodes[name]->val = carg;

            // codegening each of the root nodes in the cargs
            for (auto& [name, node] : carg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;
            // codegening each of the non-root nodes in the cargs (If these can't be codegened, the provided cargs are incomplete)
            // but it is still possible for non-root nodes to get their values from root nodes which is why this happens second
            for (auto& [name, node] : carg_nodes)
                if (!node->is_root && codegen_root_arg(scope, *node))
                    return error::general("The provided cargs are insufficient for an entry point.  Unable to deduce carg %", name);

            // Putting the cargs (in order) onto the stack
            for (const auto& name : carg_stack)
            {
                TypeRef type = type_manager->duplicate(
                    TypeInfo::Category::CONST,
                    [this, name](Scope& scope) -> bool {
                        Scope::StackVar var;
                        return
                            scope.at(name, var) ||
                            body->add_instruction(instruction::Dup(ast_def->node_info, var.ptr)) ||
                            scope.push();
                    }, carg_nodes.at(name)->val);
                if (!type ||
                    carg_nodes.at(name)->val->codegen(scope) ||
                    scope.add(name, type, ast_def->node_info)
                    ) return true;
            }

            // Codegening each of the vargs into a DLTYPE object
            for (const AstArgDecl& varg : ast_def->signature.vargs)
            {
                TypeRef ret;
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, *varg.type_expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::DLTYPE)
                    return error::compiler(varg.node_info, "Varg type did not resolve to a tensor");
                if (ret->type_dltype.tensor->codegen(scope))
                    return true;
            }

            // Resolving the procedure name of the top level def
            std::string name;
            size_t name_addr;
            if (proc_name_def(name, {}, *ast_def) ||
                bc->add_static_ref(ast_def->node_info, name, name_addr)
                ) return true;

            // Calling the def
            if (body->add_instruction(instruction::Nul(ast_def->node_info)) ||
                body->add_instruction(instruction::New(ast_def->node_info, name_addr)) ||
                body->add_instruction(instruction::Call(ast_def->node_info))
                ) return true;
            // Popping the args off the stack
            for (size_t i = 0; i < carg_stack.size() + varg_stack.size() + 1; i++)
                if (body->add_instruction(instruction::Pop(ast_def->node_info, 0)))
                    return true;
            return false;
        }

        bool FnCall::init(const AstFn& sig)
        {
            ast_fn = &sig;
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;  // copying out the old namespace
            cg_ns = sig_ns;  // replacing it with the namespace that the sig was defined in

            // Building the nodes for the cargs
            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);

                ValNode* node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;

                if (arg_decl.default_expr)
                {
                    if (arg_decl.is_packed)
                        return error::compiler(arg_decl.node_info, "Packed arguments cannot have default values");
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(init_scope, *arg_decl.default_expr, node->val_arg.default_type))
                        return true;
                }
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                carg_nodes[arg_decl.var_name] = node;
                carg_stack.push_back(arg_decl.var_name);
            }

            // Building the nodes for the vargs
            for (const auto& arg_decl : sig.signature.vargs)
            {
                // There shouldn't be any name collisions between all args
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' between cargs and vargs", arg_decl.var_name);
                if (varg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in vargs", arg_decl.var_name);

                ValNode* node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                if (arg_decl.default_expr)
                    return error::compiler(arg_decl.default_expr->node_info, "Default arguments are not allowed in vargs");
                node->val_arg.name = arg_decl.var_name;
                node->node_info = &arg_decl.node_info;
                varg_nodes[arg_decl.var_name] = node;
                varg_stack.push_back(arg_decl.var_name);
            }

            for (const AstExpr& expr : sig.signature.rets)
            {
                TypeRef ret;
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(init_scope, expr, ret))
                    return true;
                if (ret->ty == TypeInfo::Type::DLTYPE)
                {
                    // The null codegen callback will get overwritten during codegen
                    ret = type_manager->create_tensor(TypeInfo::Category::DEFAULT, nullptr);
                    if (!ret)
                        return true;
                }
                else if (ret->ty == TypeInfo::Type::TYPE)
                {
                    // Overwritting the category of the return type to const.
                    // Similar to DLTYPE, the null codegen callback will get overwritten during codegen
                    ret = type_manager->duplicate(TypeInfo::Category::CONST, nullptr, ret);
                    if (!ret)
                        return true;
                }
                else
                    return error::compiler(expr.node_info, "Return values from a function must be types");
                rets.push_back(ret);
            }

            cg_ns = old_ns;  // Putting the old namespace back
            return false;
        }

        bool FnCall::apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs)
        {
            // Checking to make sure all the provided cargs are in the signature
            for (const auto& [name, type] : cargs)
                if (!carg_nodes.contains(name))
                    return error::compiler(node_info, "The provided carg '%' does not exist in the signature", name);
            // Checking to make sure all the vargs are provided
            if (vargs.size() < varg_stack.size())
                return error::compiler(node_info, "Too few vargs provided in call");
            if (vargs.size() > varg_stack.size())
                return error::compiler(node_info, "Too many vargs provided in call");
            assert(vargs.size() == varg_stack.size());

            // Putting the carg exprs into the nodes
            for (const auto& [name, expr] : cargs)
                carg_nodes[name]->val = expr;

            // Putting the varg exprs into the nodes
            for (size_t i = 0; i < vargs.size(); i++)
                varg_nodes[varg_stack[i]]->val = vargs[i];
            return false;
        }

        bool FnCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;
            for (auto& [name, node] : varg_nodes)
                if (node->is_root && codegen_root_arg(scope, *node))
                    return true;

            // codegening each of the cargs onto the stack
            for (const std::string& name : carg_stack)
            {
                const ValNode& node = *carg_nodes.at(name);
                TypeRef val;
                if (node.val)
                    val = node.val;
                else if (node.val_arg.default_type)
                    val = node.val_arg.default_type;
                else
                    return error::compiler(*node.node_info, "Unable to deduce a value for carg '%'", name);

                if (val->codegen(scope))
                    return true;
            }

            // codegening each of the vargs onto the stack
            for (const std::string& name : varg_stack)
            {
                const ValNode& node = *varg_nodes.at(name);
                if (!node.val)
                    return error::compiler(*node.node_info, "Missing required varg '%'", name);

                if (node.val->codegen(scope))
                    return true;
            }

            // calling the procedure
            std::string proc_name;
            size_t proc_addr;
            Scope::StackVar var;
            if (proc_name_fn(proc_name, sig_ns, *ast_fn) ||
                bc->add_static_ref(*node_info, proc_name, proc_addr) ||
                scope.at("~block", var) ||
                body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                body->add_instruction(instruction::New(*node_info, proc_addr)) ||
                body->add_instruction(instruction::Call(*node_info))
                ) return true;

            // cleaning the stack (+1 from ~block), and marking the return values to the stack
            for (size_t i = 0; i < carg_stack.size() + varg_stack.size() + 1; i++)
                if (body->add_instruction(instruction::Pop(*node_info, this->rets.size())))
                    return true;
            if (scope.pop(carg_stack.size() + varg_stack.size()))
                return true;
            for (TypeRef& ret : this->rets)
            {
                // In a fn, I don't nessisarily know the ret type, so get the type from the init
                // and swap out the codegen function for a stack variable reference
                std::string var_name = scope.generate_var_name();
                CodegenCallback codegen = [node_info{ node_info }, var_name](Scope& scope) -> bool {
                    Scope::StackVar var;
                    if (scope.at(var_name, var))
                        return error::compiler(*node_info, "Unable to resolve identifier %", var_name);
                    return
                        body->add_instruction(instruction::Dup(*node_info, var.ptr)) ||
                        scope.push();
                };
                TypeRef ret_type;
                if (ret->ty == TypeInfo::Type::TYPE)
                    ret_type = type_manager->duplicate(TypeInfo::Category::CONST, codegen, ret->type_type.base);
                else
                {
                    assert(ret->ty == TypeInfo::Type::DLTYPE);
                    ret_type = type_manager->create_tensor(TypeInfo::Category::DEFAULT, codegen);
                }
                if (!ret_type)
                    return true;
                rets.push_back(ret_type);
                if (scope.push() || scope.add(var_name, ret, *node_info))
                    return true;
            }
            return false;
        }

        ModuleInfo::ModuleInfo() {}

        void ModuleInfo::init(TypeManager* pmanager)
        {
            type_manager = pmanager;
        }

        bool ModuleInfo::entry_setup(const std::string& name, const std::map<std::string, TypeRef>& cargs)
        {
            if (bc->has_proc("main"))
                return error::general("Unable to generate an entry point to a module that already has an entry point");

            body_type = BodyType::DEF;
            body_sig.block_sig = nullptr;  // This shouldn't ever get used

            cg_ns = {};
            Scope scope{ nullptr };
            AstExpr lookup_expr;
            new (&lookup_expr.expr_string) std::string();
            lookup_expr.ty = ExprType::VAR;
            lookup_expr.expr_string = name;
            TypeRef ret;
            if (codegen_expr_callee_var(scope, lookup_expr, ret))
                return true;
            assert(ret->ty == TypeInfo::Type::LOOKUP);  // With an empty scope, it should be impossible to shadow the name
            if (ret->type_lookup.lookup.defs.size() == 0)
                return error::general("Unable to find a def % entry point", name);
            if (ret->type_lookup.lookup.defs.size() > 1)
                return error::general("def % was overloaded multiple times.  Entry point is ambiguous", name);
            assert(ret->type_lookup.lookup.defs.size() == 1);
            const AstBlock& model = *ret->type_lookup.lookup.defs[0];
            ByteCodeBody main_body{ model.node_info };
            body = &main_body;

            std::unique_ptr<DefCall> call = std::make_unique<DefCall>(cg_ns);
            return
                call->init(model) ||
                call->codegen_entrypoint(scope, cargs) ||
                body->add_instruction(instruction::Ret(model.node_info)) ||
                bc->add_block("main", main_body);
        }

        TypeRef ModuleInfo::create_bool(BoolObj val)
        {
            size_t addr;
            if (bc->add_obj_bool(addr, val))
                return TypeInfo::null;
            return type_manager->create_bool(
                TypeInfo::Category::CONST,
                [node_info{ node_info }, addr](Scope& scope) -> bool {
                    return
                        body->add_instruction(instruction::New(node_info, addr)) ||
                        scope.push();
                });
        }

        TypeRef ModuleInfo::create_fty(FtyObj val)
        {
            size_t addr;
            if (bc->add_obj_fty(addr, val))
                return TypeInfo::null;
            return type_manager->create_fty(
                TypeInfo::Category::CONST,
                [node_info{ &node_info }, addr](Scope& scope) -> bool {
                    return
                        body->add_instruction(instruction::New(*node_info, addr)) ||
                        scope.push();
                });
        }

        TypeRef ModuleInfo::create_int(IntObj val)
        {
            size_t addr;
            if (bc->add_obj_int(addr, val))
                return TypeInfo::null;
            return type_manager->create_int(
                TypeInfo::Category::CONST,
                [node_info{ &node_info }, addr](Scope& scope) -> bool {
                    return
                        body->add_instruction(instruction::New(*node_info, addr)) ||
                        scope.push();
                });
        }

        TypeRef ModuleInfo::create_float(FloatObj val)
        {
            size_t addr;
            if (bc->add_obj_float(addr, val))
                return TypeInfo::null;
            return type_manager->create_float(
                TypeInfo::Category::CONST,
                [node_info{ &node_info }, addr](Scope& scope) -> bool {
                    return
                        body->add_instruction(instruction::New(*node_info, addr)) ||
                        scope.push();
                });
        }

        TypeRef ModuleInfo::create_str(const StrObj& val)
        {
            size_t addr;
            if (bc->add_obj_str(addr, val))
                return TypeInfo::null;
            return type_manager->create_string(
                TypeInfo::Category::CONST,
                [node_info{ &node_info }, addr](Scope& scope) -> bool {
                    return
                        body->add_instruction(instruction::New(*node_info, addr)) ||
                        scope.push();
                });
        }

        TypeRef ModuleInfo::create_array(const std::vector<TypeRef>& val)
        {
            if (val.size() == 0)
            {
                TypeRef tmp = type_manager->create_placeholder();
                if (!tmp)
                    return TypeInfo::null;
                return type_manager->create_array(
                    TypeInfo::Category::CONST,
                    [node_info{ &node_info }](Scope& scope) -> bool {
                        return
                            body->add_instruction(instruction::Agg(*node_info, 0)) ||
                            scope.push();
                    }, tmp);
            }
            for (size_t i = 1; i < val.size(); i++)
                if (val.front() != val[i])
                {
                    error::general("Found different types in array initialization");
                    return TypeInfo::null;
                }
            TypeRef tmp = type_manager->duplicate(TypeInfo::Category::VIRTUAL, nullptr, val.front());
            if (!tmp)
                return TypeInfo::null;
            return type_manager->create_array(
                TypeInfo::Category::CONST,
                [node_info{ &node_info }, val](Scope& scope) -> bool {
                    for (TypeRef elem : val)
                        if (elem->codegen(scope))
                            return true;
                    return
                        body->add_instruction(instruction::Agg(*node_info, val.size())) ||
                        scope.pop(val.size() - 1);
                }, tmp);
        }

        std::string label_prefix(const AstNodeInfo& type)
        {
            char buf[128];
            static size_t lbl_count = 0;
            sprintf(buf, "l%u_%uc%u_%u_%zu_", type.line_start, type.line_end, type.col_start, type.col_end, lbl_count++);
            return buf;
        }

        template<class allowed>
        bool codegen_expr_single_ret(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            std::vector<TypeRef> rets;
            if (codegen_expr(scope, expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(expr.node_info, "Expected a single value, recieved % values", rets.size());
            if (!allowed::has_default && rets[0]->cat == TypeInfo::Category::DEFAULT)
                return error::compiler(expr.node_info, "Expected a non-default value");
            if (!allowed::has_const && rets[0]->cat == TypeInfo::Category::CONST)
                return error::compiler(expr.node_info, "Expected a non-constant value");
            if (!allowed::has_ref && rets[0]->cat == TypeInfo::Category::REF)
                return error::compiler(expr.node_info, "Expected a non-reference value");
            if (!allowed::has_virtual && rets[0]->cat == TypeInfo::Category::VIRTUAL)
            {
                // See if the ret can be disambiguated to a 
                return error::compiler(expr.node_info, "Expected a non-virtual value");
            }
            ret = rets[0];
            return false;
        }

        template<class allowed>
        bool codegen_expr_multi_ret(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            if (codegen_expr(scope, expr, rets))
                return true;
            if (!allowed::has_default)
                for (const auto& ret : rets)
                    if (ret->cat == TypeInfo::Category::DEFAULT)
                        return error::compiler(expr.node_info, "Expected a non-default value");
            if (!allowed::has_const)
                for (const auto& ret : rets)
                    if (ret->cat == TypeInfo::Category::CONST)
                        return error::compiler(expr.node_info, "Expected a non-constant value");
            if (!allowed::has_ref)
                for (const auto& ret : rets)
                    if (ret->cat == TypeInfo::Category::REF)
                        return error::compiler(expr.node_info, "Expected a non-reference value");
            if (!allowed::has_virtual)
                for (const auto& ret : rets)
                    if (ret->cat == TypeInfo::Category::VIRTUAL)
                        return error::compiler(expr.node_info, "Expected a non-virtual value");
            return false;
        }

        bool arg_type(TypeRef& type, Scope& scope, const AstArgDecl& arg)
        {
            TypeRef explicit_type;
            if (arg.type_expr)
            {
                CodegenCallback codegen = [&arg](Scope& scope) -> bool {
                    Scope::StackVar var;
                    return
                        scope.at(arg.var_name, var) ||
                        body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                        scope.push();
                };
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, *arg.type_expr, explicit_type))
                    return true;
                if (explicit_type->ty == TypeInfo::Type::DLTYPE)
                {
                    if (body_type == BodyType::INTR)
                        explicit_type = type_manager->create_edge(TypeInfo::Category::DEFAULT, codegen);
                    else
                        explicit_type = type_manager->create_tensor(TypeInfo::Category::CONST, codegen);
                    if (!explicit_type)
                        return true;
                }
                else if (explicit_type->ty == TypeInfo::Type::TYPE)
                    explicit_type = type_manager->duplicate(TypeInfo::Category::CONST, codegen, explicit_type->type_type.base);
                else
                    return error::compiler(arg.type_expr->node_info, "Type expression for parameter % did not resolve to a type.", arg.var_name);
                
                if (arg.is_packed)
                {
                    explicit_type = type_manager->create_array(TypeInfo::Category::CONST, codegen, explicit_type);
                    if (!explicit_type)
                        return true;
                }
            }
            
            TypeRef default_type;
            if (arg.default_expr)
            {
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *arg.default_expr, default_type))
                    return true;
            }

            if (!explicit_type)
            {
                if (!default_type)
                    return error::compiler(arg.node_info, "Missing both the type expression and the default expression");
                // only the default type was specified
                type = default_type;
                return false;
            }
            if (!default_type)
            {
                // only the explicit type was specified
                type = explicit_type;
                return false;
            }
            // both the explicit and default type were specified
            if (explicit_type != default_type)
                return error::compiler(arg.node_info, "The default value's type for argument '%' did not match the explicitly declared type", arg.var_name);
            type = explicit_type;
            return false;
        }

        bool match_elems(const std::vector<AstExpr>& lhs, const std::vector<TypeRef>& rhs, CodegenCallback& setup_fn, std::vector<CodegenCallback>& elem_fn)
        {
            for (const auto& expr : lhs)
            {
                switch (expr.ty)
                {
                case ExprType::VAR:
                    ;
                case ExprType::UNARY_UNPACK:
                    ;
                case ExprType::VAR_DECL:
                    ;
                }
            }
            return true;
        }

        bool match_carg_sig(Scope& scope, Scope& sig_scope,
            const std::vector<AstArgDecl>& sig, const std::vector<AstExpr>& args,
            std::map<std::string, TypeRef>& cargs)
        {
            // Codegening the signature cargs to compare types against the call arguments
            // If the signature contains a variable naming conflict, it'll be caught during the scope add operation
            // This ensures that during the remaining logic, the signature will be valid
            for (const AstArgDecl& carg : sig)
            {
                TypeRef ret;
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(sig_scope, *carg.type_expr, ret) ||
                    sig_scope.push() ||
                    sig_scope.add(carg.var_name, ret->type_type.base, carg.node_info)
                    ) return true;
            }

            auto sig_it = sig.begin();
            for (auto arg_it = args.begin(); arg_it != args.end(); arg_it++)
            {
                if (arg_it->ty == ExprType::BINARY_ASSIGN)
                {
                    // Its a keyword argument, ez clap
                    if (arg_it->expr_binary.left->ty != ExprType::VAR)
                        return error::compiler(arg_it->expr_binary.left->node_info, "The lhs of a keyword argument must be an identifier");
                    if (sig_it->is_packed)
                        return error::compiler(arg_it->node_info, "Assigning packed arguments via a keyword isn't allowed");
                    // Finding which signature element matches the keyword argument
                    for (auto it = sig.begin(); it != sig.end(); it++)
                        if (it->var_name == arg_it->expr_binary.left->expr_string)
                        {
                            assert(sig_scope.contains(it->var_name));
                            // Checking if the argument has already been mapped
                            if (cargs.contains(it->var_name))
                                return error::compiler(arg_it->node_info, "Found multiple values for argument '%'", it->var_name);
                            Scope::StackVar var;
                            // its a valid keyword argument, map it and move the position of sig_it to it + 1
                            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *arg_it->expr_binary.right, cargs[it->var_name]) ||
                                sig_scope.at(it->var_name, var) ||
                                var.type != cargs.at(it->var_name)
                                ) return true;
                            sig_it = it + 1;
                            break;
                        }
                }
                else
                {
                    // Codegen whatever the argument is and see if its able to match the signature
                    std::vector<TypeRef> rets;
                    if (codegen_expr(scope, *arg_it, rets))
                        return true;
                    if (rets.size() == 0)
                        continue;  // That value didn't matter I guess
                    auto rets_it = rets.begin();
                    size_t unpack_min_sz = 0;
                    if (!sig_it->is_packed)
                    {
                        // The signature element isn't packed
                        if ((*rets_it)->ty == TypeInfo::Type::UNPACK)
                        {
                            // TODO: figure out argument unpacking in signatures
                            // This case should place a runtime requirement on the length of the underlying array
                            return error::compiler(arg_it->node_info, "Internal error: not implemented");
                        }
                        else
                        {
                            // Straight forward case, eat the current ret and increment the iterator
                            // First check if the argument has already been provided a value
                            if (cargs.contains(sig_it->var_name))
                                return error::compiler(arg_it->node_info, "Found multiple values for argument '%'", sig_it->var_name);
                            cargs[sig_it->var_name] = *rets_it;
                            rets_it++;
                            if (rets_it == rets.end())
                            {
                                // Finished eating all the rets from the passed value, move onto the next one.
                                // continue might not be right here...
                                // Future me responding: nope it wasn't, I needed to also increment sig_it.
                                sig_it++;
                                continue;
                            }
                        }
                    }
                    else
                    {
                        // The signature element is packed, in this case we keep eating the arguments until either
                        // an unpacked value or a type doesn't match.  If it terminates with an unpacked value,
                        // the packed signature argument doesn't keep eating.  This means that we can't
                        // peel elements of the array off the back of it, only the front,
                        // but it allows for much cleaner rule around multiple packed arguments in a signature.
                        // This way here, a single unpacked value won't eat multiple packed signature arguments
                        // To eat multiple packed signature arguments, you need multiple unpacked values.
                        // This results in much more intuitive behaviour where there's a one to one mapping between
                        // the passed arguments and the signature as it appears in code.
                        
                        // First up, make sure the argument hasn't already been given a value
                        if (cargs.contains(sig_it->var_name))
                            return error::compiler(arg_it->node_info, "Found multiple values for argument '%'", sig_it->var_name);
                        // Next get the signature argument type to match the passed values against, and get ready
                        // for the bullshit that's gonna happen next
                        Scope::StackVar var;
                        TypeRef ety;
                        if (sig_scope.at(sig_it->var_name, var) ||
                            var.type->to_obj(sig_it->node_info, ety)
                            ) return true;
                        CodegenCallback cb = [&node_info{ arg_it->node_info }](Scope& scope) {
                            return body->add_instruction(instruction::Agg(node_info, 0));
                        };
                        while (true)
                        {
                            if (rets_it == rets.end())
                            {
                                // Hit the end of the rets recieved from the current arg_it and we haven't hit
                                // neither a value that didn't match the signature, nor an unpacked value.
                                // codegen the next passed value and continue
                                arg_it++;
                                if (arg_it == args.end())
                                {
                                    // The packed signature element ate all the remaining cargs given
                                    TypeRef ret = type_manager->create_array(TypeInfo::Category::CONST, cb, var.type);
                                    if (!ret)
                                        return true;
                                    cargs[sig_it->var_name] = ret;
                                    return false;  // Successfully matched the args to the signature!
                                }
                                // First check if the next arg is a keyword argument.  If it is, the current
                                // carg is finished and we need to resume the normal algorithm
                                if (arg_it->ty == ExprType::BINARY_ASSIGN)
                                {
                                    // Its a keyword argument, save the current argument then decrement arg_it
                                    // so that when the for loop increments it, its back to the current position
                                    TypeRef ret = type_manager->create_array(TypeInfo::Category::CONST, cb, var.type);
                                    if (!ret)
                                        return true;
                                    cargs[sig_it->var_name] = ret;
                                    arg_it--;
                                    break;  // breaking the while (true), not the for loop
                                }
                                // We didn't hit the last arg yet, so codegen it into rets and reset rets_it
                                rets.clear();
                                if (codegen_expr(scope, *arg_it, rets))
                                    return true;
                                rets_it = rets.begin();
                                // Continue on as though nothing happend
                            }

                            if ((*rets_it)->ty != TypeInfo::Type::UNPACK)
                            {
                                // Its a standard argument also it can't contain any unpacks since capturing an unpack
                                // is not allowed
                                if ((*rets_it)->cat == TypeInfo::Category::VIRTUAL)
                                    return error::compiler(arg_it->node_info, "Arguments must be non-virtual");
                                if (var.type != *rets_it)
                                {
                                    // Found an argument which doesn't match the type of the packed signature element
                                    // This terminates the current argument, but leaves leftover passed values in rets
                                    // which could potentially come from half an expanded tuple.  This means that we
                                    // can't leverage the previous code for this.
                                    error::pop_last(); // getting rid of the error codegen_expr_attempt_implicit generated
                                    TypeRef ret = type_manager->create_array(TypeInfo::Category::CONST, cb, var.type);
                                    if (!ret)
                                        return true;
                                    cargs[sig_it->var_name] = ret;
                                    // The rest of rets need to be matched against the remaining signature.  What makes this
                                    // difficult is what happens when we encounter a packed signature element during this
                                    for (; rets_it == rets.end(); rets_it++)
                                    {
                                        // This for loop iterates over the direct matches between the passed arguments
                                        // and non-packed signature elements.
                                        sig_it++;
                                        if (sig_it == sig.end())
                                            return error::compiler(arg_it->node_info, "Too many arguments were provided");
                                        if (cargs.contains(sig_it->var_name))
                                            return error::compiler(arg_it->node_info, "Found multiple values for argument '%'", sig_it->var_name);
                                        // Updating the current info kept about the signature element
                                        if (sig_scope.at(sig_it->var_name, var) ||
                                            var.type->to_obj(sig_it->node_info, ety)
                                            ) return true;
                                        // If the signature element is a packed value, I can directly leverage the outer
                                        // while (true) loop to parse out the rest of the signature
                                        if (sig_it->is_packed)
                                            break;

                                        // The signature element wasn't packed, it's just a simple match
                                        if (var.type != *rets_it)
                                            return true;
                                        cargs[sig_it->var_name] = *rets_it;
                                    }
                                    if (rets_it != rets.end())
                                        // Didn't hit the end of rets, instead there was a packed signature element
                                        continue;
                                    // Finished rets cleanly, break out of the while (true) and resume the normal algorithm
                                    break;
                                }
                                cb = [lhs{ cb }, rhs{ (*rets_it)->codegen }, ety, &node_info{ arg_it->node_info }](Scope& scope) {
                                    return
                                        lhs(scope) ||
                                        rhs(scope) ||
                                        ety->codegen(scope) ||
                                        body->add_instruction(instruction::Arr(node_info)) ||
                                        body->add_instruction(instruction::Add(node_info));
                                };
                            }
                            else
                            {
                                if ((*rets_it)->type_array.elem == var.type)
                                {
                                    // Perfect match between the unpacked passed value and the signature type
                                    // This means that the lhs and rhs can directly be added
                                    TypeRef ret = type_manager->create_array(
                                        TypeInfo::Category::CONST,
                                        [lhs{ cb }, rhs{ (*rets_it)->codegen }, ety, &node_info{ arg_it->node_info }](Scope& scope) {
                                        return
                                            lhs(scope) ||
                                            rhs(scope) ||
                                            ety->codegen(scope) ||
                                            body->add_instruction(instruction::Arr(node_info)) ||
                                            body->add_instruction(instruction::Add(node_info));
                                    },
                                        var.type);
                                    if (!ret)
                                        return true;
                                    cargs[sig_it->var_name] = ret;
                                    assert(rets_it + 1 == rets.end());
                                    // Finished the packed signature argument, resume the normal algorithm
                                    break;  // breaking the while (true), not the for loop
                                }

                                // TODO: implicitly cast the unpacked values
                                return error::compiler(arg_it->node_info, "Internal error: implicit unpack cast has not been implemented");
                            }
                        }
                    }
                }
            }
            return false;
        }

        bool match_def_sig(Scope& scope, Scope& sig_scope, const AstBlock& def,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs)
        {
            // If the cargs don't fit, you must acquit
            if (match_carg_sig(scope, sig_scope, def.signature.cargs, param_cargs, cargs))
                return true;
            
            for (const AstArgDecl& varg : def.signature.vargs)
                if (varg.is_packed)
                    return error::compiler(varg.node_info, "Internal error: packed vargs has not been implemented");

            if (def.signature.vargs.size() == vargs.size())
            {
                vargs = param_vargs;
                return false;
            }
            return true;
        }

        bool match_intr_sig(Scope& scope, Scope& sig_scope, const AstBlock& intr,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs)
        {
            if (match_carg_sig(scope, sig_scope, intr.signature.cargs, param_cargs, cargs))
                return true;

            for (const AstArgDecl& varg : intr.signature.vargs)
                if (varg.is_packed)
                    return error::compiler(varg.node_info, "Internal error: packed vargs has not been implemented");

            if (intr.signature.vargs.size() == vargs.size())
            {
                vargs = param_vargs;
                return false;
            }
            return true;
        }

        bool match_fn_sig(Scope& scope, Scope& sig_scope, const AstFn& fn,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs)
        {
            if (match_carg_sig(scope, sig_scope, fn.signature.cargs, param_cargs, cargs))
                return true;

            for (const AstArgDecl& varg : fn.signature.vargs)
                if (varg.is_packed)
                    return error::compiler(varg.node_info, "Internal error: packed vargs has not been implemented");

            if (fn.signature.vargs.size() != param_vargs.size())
                return true;

            for (size_t i = 0; i < param_vargs.size(); i++)
            {
                TypeRef type;
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(sig_scope, *fn.signature.vargs[i].type_expr, type))
                    return true;
                if (type->ty == TypeInfo::Type::TYPE)
                {
                    if (type->type_type.base != param_vargs[i])
                        return true;
                    continue;
                }
                if (type->ty == TypeInfo::Type::DLTYPE)
                    return error::compiler(fn.signature.vargs[i].node_info, "Internal error: not implemented");
                return error::compiler(fn.signature.vargs[i].node_info, "Type expression for varg did not resolve to a type");
            }

            vargs = param_vargs;
            return false;
        }

        bool codegen_expr_callee_kw(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            size_t addr;
            switch (expr.expr_kw)
            {
            case ExprKW::F16:
                if (bc->add_obj_fty(addr, core::EdgeFty::F16))
                    return true;
                break;
            case ExprKW::F32:
                if (bc->add_obj_fty(addr, core::EdgeFty::F32))
                    return true;
                break;
            case ExprKW::F64:
                if (bc->add_obj_fty(addr, core::EdgeFty::F64))
                    return true;
                break;
            }
            ret = type_manager->create_fty(
                TypeInfo::Category::CONST,
                [&expr, addr](Scope& scope) {
                    return body->add_instruction(instruction::New(expr.node_info, addr));
                });
            return !ret;
        }

        bool codegen_expr_callee_var(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            Scope::StackVar var;
            if (!scope.at(expr.expr_string, var))
            {
                ret = var.type;
                return false;
            }

            // Couldn't find it in the scope, look for it in the code module
            CodeModule::LookupResult lookup;
            if (mod->lookup({ mod->root, cg_ns.begin(), cg_ns.end() }, expr.expr_string, lookup))
            {
                // the identifier didn't match anything in the scope nor the code module.
                // This makes it an unresolved identifier.
                // Note: implicit identifiers would cause this to execute, but that case is handled in the
                // codegen_expr_assign function.  So virtual values are not needed here, its just an error
                return error::compiler(expr.node_info, "Unresolved identifier: %", expr.expr_string);
            }

            // returning the attrs as a virtual type
            ret = type_manager->create_lookup(expr.expr_string, std::move(lookup));
            return !ret;
        }

        bool codegen_expr_callee_dot(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            std::vector<TypeRef> rets;
            if (codegen_expr_dot(scope, expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(expr.node_info, "Expected a single value");
            ret = rets[0];
            return false;
        }

        bool codegen_expr_callee_cargs(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            // Don't need to worry about special cases like I do in codegen_expr_cargs
            // since the result of this function will be fed into codegen_expr_vargs,
            // all the special cases would be in valid.
            // So instead, I can directly codegen the callee and switch based on that result
            TypeRef callee;
            if (codegen_expr_callee(scope, *expr.expr_call.callee, callee))
                return true;
            // Only lookup results and def/fn refs are valid
            // Since def/fn refs haven't been implemented, only lookups are valid
            if (callee->ty != TypeInfo::Type::LOOKUP)
                return error::compiler(expr.node_info, "Expected a def or fn as the lhs of a carg");
            // Bind the cargs to the lookup and allow the codegen_expr_vargs to resolve the proc.
            // This is nessisary since overloading based on vargs is allowed
            ret = type_manager->create_cargbind(callee->type_lookup.lookup, expr.expr_call.args);
            return !ret;
        }

        bool codegen_expr_callee(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            switch (expr.ty)
            {
            case ExprType::KW:  // codegen_expr_callee is called by codegen_expr_cargs for the tensor case
                return codegen_expr_callee_kw(scope, expr, ret);
            case ExprType::VAR:
                return codegen_expr_callee_var(scope, expr, ret);
            case ExprType::DOT:
                return codegen_expr_callee_dot(scope, expr, ret);
            case ExprType::CARGS_CALL:
                return codegen_expr_callee_cargs(scope, expr, ret);
            default:
                return codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr, ret);
            }
        }

        bool codegen_expr_bool(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef ret = type_manager->create_bool(
                TypeInfo::Category::CONST,
                [&expr](Scope& scope) {
                    size_t addr;
                    return
                        bc->add_obj_bool(addr, expr.expr_bool) ||
                        body->add_instruction(instruction::New(expr.node_info, addr)) ||
                        scope.push();
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_int(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef ret = type_manager->create_int(
                TypeInfo::Category::CONST,
                [&expr](Scope& scope) {
                    size_t addr;
                    return
                        bc->add_obj_int(addr, expr.expr_int) ||
                        body->add_instruction(instruction::New(expr.node_info, addr)) ||
                        scope.push();
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_float(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef ret = type_manager->create_float(
                TypeInfo::Category::CONST,
                [&expr](Scope& scope) {
                    size_t addr;
                    return
                        bc->add_obj_float(addr, expr.expr_float) ||
                        body->add_instruction(instruction::New(expr.node_info, addr)) ||
                        scope.push();
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_string(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            size_t addr;
            if (bc->add_obj_str(addr, expr.expr_string))
                return true;
            TypeRef ret = type_manager->create_string(
                TypeInfo::Category::CONST,
                [&expr, addr](Scope& scope) {
                    return
                        body->add_instruction(instruction::New(expr.node_info, addr)) ||
                        scope.push();
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_array(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            if (expr.expr_agg.elems.size() == 0)
            {
                TypeRef tmp = type_manager->create_placeholder();
                if (!tmp)
                    return true;
                TypeRef ret = type_manager->create_array(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        return
                            body->add_instruction(instruction::Agg(expr.node_info, 0)) ||
                            scope.push();
                    }, tmp);
                if (!ret)
                    return true;
                rets.push_back(ret);
                return false;
            }

            size_t sz = 0;
            std::vector<TypeRef> elem_types;
            for (const auto& elem_expr : expr.expr_agg.elems)
            {
                if (codegen_expr(scope, elem_expr, elem_types))
                    return true;
                sz = elem_types.size();
            }
            
            TypeRef type = type_manager->duplicate(TypeInfo::Category::VIRTUAL, nullptr, elem_types.front());
            for (TypeRef elem_type : elem_types)
                if (type != elem_type)
                    return error::compiler(expr.node_info, "Unable to construct an array literal given different types - % and %", type, elem_type);

            TypeRef ret = type_manager->create_array(
                TypeInfo::Category::CONST,
                [&expr, elem_types](Scope& scope) -> bool {
                    for (TypeRef elem_type : elem_types)
                        if (elem_type->codegen(scope))  // don't need to copy since its const
                            return true;
                    if (body->add_instruction(instruction::Agg(expr.node_info, elem_types.size())))
                        return true;
                    assert(elem_types.size() != 0);
                    return scope.pop(elem_types.size() - 1);
                }, type);
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_tuple(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> elem_types;
            for (const auto& elem_expr : expr.expr_agg.elems)
                if (codegen_expr(scope, elem_expr, elem_types))
                    return true;
            TypeRef ret = type_manager->create_tuple(
                TypeInfo::Category::CONST,
                [&expr, elem_types](Scope& scope) {
                    for (const auto& type : elem_types)
                        type->codegen(scope);
                    return
                        body->add_instruction(instruction::Agg(expr.node_info, elem_types.size())) ||
                        scope.pop(elem_types.size() - 1);  // -1 for the added tuple
                }, elem_types);
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_pos(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> results;
            if (codegen_expr(scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The positive operator must be provided with a single numeric type, recieved '%' arguments", results.size());
            if (results[0]->ty != TypeInfo::Type::INT && results[0]->ty != TypeInfo::Type::FLOAT)
                return error::compiler(expr.node_info, "The positive operator must be provided with a single numeric type, recieved a '%' type", results[0]->to_string());
            rets.push_back(results[0]);
            return false;  // the positive operator is a no-op
        }

        bool codegen_expr_neg(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> results;
            if (codegen_expr(scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The negative operator must be provided with a single numeric type, recieved '%' arguments", results.size());
            if (results[0]->ty != TypeInfo::Type::INT && results[0]->ty != TypeInfo::Type::FLOAT)
                return error::compiler(expr.node_info, "The negative operator must be provided with a single numeric type, recieved a '%' type", results[0]->to_string());
            // TODO: implement a neg instruction
            return error::compiler(expr.node_info, "Internal error: I forgot to implement a neg instruction");
        }

        bool codegen_expr_not(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> results;
            if (codegen_expr(scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The not operator must be provided with a single boolean type, recieved '%' arguments", results.size());
            if (results[0]->ty != TypeInfo::Type::BOOL)
                return error::compiler(expr.node_info, "The not operator must be provided with a single boolean type, recieved a '%' type", results[0]->to_string());
            // TODO: implement a not instruction
            return error::compiler(expr.node_info, "Internal error: I forgot to implement a not instruction");
        }
        
        bool codegen_expr_unpack(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> args;
            if (codegen_expr_multi_ret<TypeInfo::NonVirtual>(scope, *expr.expr_unary.expr, args))
                return true;
            if (args.size() >= 2 || args.size() == 0)
            {
                // For multiple return values and no return values, unpacking is a noop
                for (auto arg : args)
                    rets.push_back(arg);
                return false;
            }
            
            // One value was given.  Need to actually do stuff now
            TypeRef arg = args[0];
            if (arg->ty == TypeInfo::Type::ARRAY)
            {
                // Array types just turn into virtual arrays cause the length of the array isn't known at compile time
                TypeRef type = type_manager->create_unpack(arg->codegen, arg->type_array.elem);
                if (!type)
                    return true;
                rets.push_back(type);
                return false;
            }
            if (arg->ty == TypeInfo::Type::TUPLE)
            {
                for (const auto& elem : arg->type_tuple.elems)
                    rets.push_back(elem);
                return false;
            }
            return error::compiler(expr.node_info, "Unable to unpack the given type");
        }

        bool codegen_expr_fwd(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef ret;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_unary.expr, ret))
                return true;
            if (ret->ty != TypeInfo::Type::TENSOR)
                return error::compiler(expr.expr_unary.expr->node_info, "Expected tensor, recieved %", ret->to_string());
            ret = type_manager->create_edge(
                TypeInfo::Category::DEFAULT,
                [&expr, ret](Scope& scope) -> bool {
                    return
                        ret->codegen(scope) ||
                        body->add_instruction(instruction::GFwd(expr.node_info));
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_bwd(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef ret;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_unary.expr, ret))
                return true;
            if (ret->ty != TypeInfo::Type::TENSOR)
                return error::compiler(expr.expr_unary.expr->node_info, "Expected tensor, recieved %", ret->to_string());
            ret = type_manager->create_edge(
                TypeInfo::Category::DEFAULT,
                [&expr, ret](Scope& scope) -> bool {
                    return
                        ret->codegen(scope) ||
                        body->add_instruction(instruction::GBwd(expr.node_info));
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        template<bool(TypeInfo::*check)() const, const char noun[], const char verb[], const char name[], class Node>
        bool codegen_expr_binop_basic(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", noun);
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", noun);

            // Checking if both the lhs and rhs are either tensors or edges, in which case
            // generate a new node or block to accomplish the operation
            if (lhs_type->ty == TypeInfo::Type::TENSOR && rhs_type->ty == TypeInfo::Type::TENSOR)
            {
                AstExpr lookup_expr;
                new (&lookup_expr.expr_string) std::string();
                lookup_expr.ty = ExprType::VAR;
                lookup_expr.node_info = expr.node_info;
                lookup_expr.expr_string = name;
                TypeRef ret;
                if (codegen_expr_callee_var(scope, lookup_expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::LOOKUP)
                    return error::compiler(expr.node_info, "def % overload is shadowed in the local scope for tensor % operation", name, noun);
                if (ret->type_lookup.lookup.defs.size() == 0)
                    return error::compiler(expr.node_info, "Unable to find a def % overload for tensor % operation", name, noun);
                if (ret->type_lookup.lookup.defs.size() > 1)
                    return error::compiler(expr.node_info, "def % was overloaded multiple times.  Tensor % operation is ambiguous", name, noun);
                assert(ret->type_lookup.lookup.defs.size() == 1);
                const AstBlock& __def__ = *ret->type_lookup.lookup.defs[0];
                if (__def__.signature.vargs.size() < 2)
                    return error::compiler(expr.node_info, "Too few vargs found in def % signature, expected exactly two", name);
                if (__def__.signature.vargs.size() > 2)
                    return error::compiler(expr.node_info, "Too many vargs found in def % signature, expected exactly two", name);
                if (__def__.signature.rets.size() != 1)
                    return error::compiler(expr.node_info, "Expected exactly one return value from def %", name);
                // For some reason on debug builds, the desruction of the DefCall object invalidates all the
                // lambdas inside the type manager resulting in a crash later on when either the block exits and
                // the type manager deletes all the types generated during the block's compilation,
                // or when another operation attempts to use a codegen method which ever comes first.
                // 
                // I tracked the bug down to a _free_dbg call on the memory allocated for the DefCall object
                // (after all the dtors have run), so its probably an MSVC compiler bug.
                // 
                // The release build works fine, but I might need to switch c++ compilers for debugging
                //std::unique_ptr<DefCall> call = std::make_unique<DefCall>(ret->type_lookup.lookup.ns);
                DefCall* call = new DefCall(ret->type_lookup.lookup.ns);
                bool ret_val =
                    call->init(__def__) ||
                    call->apply_args(expr.node_info, {}, { lhs_type, rhs_type }) ||
                    call->codegen(scope, rets);
                delete call;
                return ret_val;
            }

            if (lhs_type->ty == TypeInfo::Type::EDGE && rhs_type->ty == TypeInfo::Type::EDGE)
            {
                AstExpr lookup_expr;
                new (&lookup_expr.expr_string) std::string();
                lookup_expr.ty = ExprType::VAR;
                lookup_expr.node_info = expr.node_info;
                lookup_expr.expr_string = name;
                TypeRef ret;
                if (codegen_expr_callee_var(scope, lookup_expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::LOOKUP)
                    return error::compiler(expr.node_info, "intr % overload is shadowed in the local scope for edge % operation", name, noun);
                if (ret->type_lookup.lookup.intrs.size() == 0)
                    return error::compiler(expr.node_info, "Unable to find a intr % overload for edge % operation", name, noun);
                if (ret->type_lookup.lookup.intrs.size() > 1)
                    return error::compiler(expr.node_info, "intr % was overloaded multiple times.  Edge % operation is ambiguous", name, noun);
                assert(ret->type_lookup.lookup.intrs.size() == 1);
                const AstBlock& __intr__ = *ret->type_lookup.lookup.intrs[0];
                if (__intr__.signature.vargs.size() < 2)
                    return error::compiler(expr.node_info, "Too few vargs found in intr % signature, expected exactly two", name);
                if (__intr__.signature.vargs.size() > 2)
                    return error::compiler(expr.node_info, "Too many vargs found in intr % signature, expected exactly two", name);
                if (__intr__.signature.rets.size() != 1)
                    return error::compiler(expr.node_info, "Expected exactly one return value from intr %", name);
                std::unique_ptr<IntrCall> call = std::make_unique<IntrCall>(ret->type_lookup.lookup.ns);
                return
                    call->init(__intr__) ||
                    call->apply_args(expr.node_info, {}, { lhs_type, rhs_type }) ||
                    call->codegen(scope, rets);
            }

            if (!((*lhs_type).*check)())
                return error::compiler(expr.node_info, "Unable to % values with type '%'", verb, lhs_type->to_string());
            if (lhs_type != rhs_type)  // Only exact matches are allowed.  Explicit casting is otherwise nessisary
                return error::compiler(expr.node_info, "Unable to % a value of type '%' with a value of type '%'",
                    verb, lhs_type->to_string(), rhs_type->to_string());

            TypeRef type;
            if (lhs_type->to_obj(expr.node_info, type))
                return true;
            TypeRef ret = type_manager->duplicate(
                TypeInfo::Category::CONST,
                [type, &expr, lhs_type, rhs_type](Scope& scope) {
                    return
                        lhs_type->codegen(scope) ||  // codegen the lhs
                        rhs_type->codegen(scope) ||  // codegen the rhs
                        type->codegen(scope) ||
                        body->add_instruction(Node(expr.node_info)) ||
                        scope.pop(2);  // pops ty, rhs, lhs; pushes ret
                }, lhs_type);
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }
        
        using op = bool(*)(Scope&, const AstExpr&, std::vector<TypeRef>&);

        // C++ doesn't like string literals as template arguments
        char add_noun[] = "addition"      ; char add_verb[] = "add"         ; char add_name[] = "__add__";
        char sub_noun[] = "subtraction"   ; char sub_verb[] = "subtract"    ; char sub_name[] = "__sub__";
        char mul_noun[] = "multiplication"; char mul_verb[] = "multiply"    ; char mul_name[] = "__mul__";
        char div_noun[] = "division"      ; char div_verb[] = "divide"      ; char div_name[] = "__div__";
        char mod_noun[] = "modulus"       ; char mod_verb[] = "modulus"     ; char mod_name[] = "__mod__";
        char pow_noun[] = "exponentiation"; char pow_verb[] = "exponentiate"; char pow_name[] = "__pow__";
        op codegen_expr_add = codegen_expr_binop_basic<&TypeInfo::check_add, add_noun, add_verb, add_name, instruction::Add>;
        op codegen_expr_sub = codegen_expr_binop_basic<&TypeInfo::check_sub, sub_noun, sub_verb, sub_name, instruction::Sub>;
        op codegen_expr_mul = codegen_expr_binop_basic<&TypeInfo::check_mul, mul_noun, mul_verb, mul_name, instruction::Mul>;
        op codegen_expr_div = codegen_expr_binop_basic<&TypeInfo::check_div, div_noun, div_verb, div_name, instruction::Div>;
        op codegen_expr_mod = codegen_expr_binop_basic<&TypeInfo::check_mod, mod_noun, mod_verb, mod_name, instruction::Mod>;
        op codegen_expr_pow = codegen_expr_binop_basic<&TypeInfo::check_pow, pow_noun, pow_verb, pow_name, instruction::Pow>;

        template<bool(TypeInfo::*check)() const, const char* noun, const char* verb, class Node>
        bool codegen_expr_binop_iop(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<TypeInfo::Mutable>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", noun);
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", noun);

            if (!((*lhs_type).*check)())
                return error::compiler(expr.node_info, "Unable to % values with type '%'", verb, lhs_type->to_string());
            if (lhs_type != rhs_type)
                return error::compiler(expr.node_info, "Unable to % a value of type '%' with a vlue of type '%'",
                    verb, lhs_type->to_string(), rhs_type->to_string());

            TypeRef ty;
            return
                lhs_type->to_obj(expr.node_info, ty) ||
                lhs_type->codegen(scope) ||
                rhs_type->codegen(scope) ||
                ty->codegen(scope) ||
                body->add_instruction(instruction::Cpy(expr.node_info)) ||
                scope.pop() ||
                ty->codegen(scope) ||
                body->add_instruction(Node(expr.node_info)) ||
                scope.pop(3);
        }

        char iadd_noun[] = "addition assignment"      ; char iadd_verb[] = "add"         ;
        char isub_noun[] = "subtraction assignment"   ; char isub_verb[] = "subtract"    ;
        char imul_noun[] = "multiplication assignment"; char imul_verb[] = "multiply"    ;
        char idiv_noun[] = "division assignment"      ; char idiv_verb[] = "divide"      ;
        char imod_noun[] = "modulus assignment"       ; char imod_verb[] = "modulus"     ;
        char ipow_noun[] = "exponentiation assignment"; char ipow_verb[] = "exponentiate";
        op codegen_expr_iadd = codegen_expr_binop_iop<&TypeInfo::check_add, iadd_noun, iadd_verb, instruction::IAdd>;
        op codegen_expr_isub = codegen_expr_binop_iop<&TypeInfo::check_sub, isub_noun, isub_verb, instruction::ISub>;
        op codegen_expr_imul = codegen_expr_binop_iop<&TypeInfo::check_mul, imul_noun, imul_verb, instruction::IMul>;
        op codegen_expr_idiv = codegen_expr_binop_iop<&TypeInfo::check_div, idiv_noun, idiv_verb, instruction::IDiv>;
        op codegen_expr_imod = codegen_expr_binop_iop<&TypeInfo::check_mod, imod_noun, imod_verb, instruction::IMod>;
        op codegen_expr_ipow = codegen_expr_binop_iop<&TypeInfo::check_pow, ipow_noun, ipow_verb, instruction::IPow>;

        bool codegen_expr_assign(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            AstExpr* lhs = expr.expr_binary.left.get();
            AstExpr* rhs = expr.expr_binary.right.get();
            bool is_const = lhs->ty == ExprType::UNARY_CONST;
            bool is_ref = lhs->ty == ExprType::UNARY_REF;
            if (is_const || is_ref)
                lhs = lhs->expr_unary.expr.get();

            if (lhs->ty == ExprType::VAR && !scope.contains(lhs->expr_string))
            {
                // Its an implicit declaration
                TypeRef val;
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *rhs, val))
                    return true;
                // doing the codegen for the right hand side
                if (val->codegen(scope))
                    return true;

                // Overwriting the codegen with a dup instruction
                CodegenCallback cb = [lhs](Scope& scope)
                {
                    Scope::StackVar stack_var;
                    return
                        scope.at(lhs->expr_string, stack_var) ||
                        body->add_instruction(instruction::Dup(lhs->node_info, stack_var.ptr)) ||
                        scope.push();
                };
                val->codegen = cb;
                // Overwriting the type category based on the modifiers
                // the non-virtual condition is handled by codegen_expr_single_ret<TypeInfo::NonVirtual>
                if (is_const)
                    val->cat = TypeInfo::Category::CONST;
                else if (is_ref)
                {
                    if (!val->in_category<TypeInfo::Mutable>())
                        return error::compiler(expr.node_info, "Unable to take a reference of an immutable object");
                    val->cat = TypeInfo::Category::REF;
                }
                else
                {
                    // objects that are handled by the graph builder
                    // don't have runtime type objects, and can't be copied
                    if (val->check_cpy())
                    {
                        // copying the object during assignment
                        TypeRef type;
                        if (val->to_obj(expr.node_info, type) ||
                            type->codegen(scope) ||
                            body->add_instruction(instruction::Cpy(expr.node_info)) ||
                            scope.pop()
                            ) return true;
                    }
                    val->cat = TypeInfo::Category::DEFAULT;
                }
                return scope.add(lhs->expr_string, val, expr.node_info);
            }
            // const / ref can only be used with implicit declarations
            else if (is_ref)
                return error::compiler(expr.node_info, "Invalid usage of keyword 'ref'");
            else if (is_const)
                return error::compiler(expr.node_info, "Invalid usage of keyword 'const'");
            else if (lhs->ty == ExprType::LIT_TUPLE)
            {
                // Multiple assignment, handle implicit and standard assignment on a per-element basis
                std::vector<TypeRef> rhs_types;
                if (codegen_expr(scope, *rhs, rhs_types))
                    return true;
                CodegenCallback setup_fn;
                std::vector<CodegenCallback> elem_fns;

                // TODO: implement match_elems
                return error::compiler(lhs->node_info, "Internal error: not implemented");
                
                if (match_elems(lhs->expr_agg.elems, rhs_types, setup_fn, elem_fns))
                    return error::compiler(expr.node_info, "Unable to match types across assignment");
                if (setup_fn(scope))
                    return true;
                for (const auto& fn : elem_fns)
                    if (fn(scope))
                        return true;
                return false;
            }
            else
            {
                // Standard assignment to a prior variable or explicit declaration
                TypeRef lhs_type, rhs_type;
                TypeRef ty;
                if (codegen_expr_single_ret<TypeInfo::Mutable>(scope, *lhs, lhs_type) ||
                    codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *rhs, rhs_type)
                    ) return true;
                
                if (rhs_type->ty == TypeInfo::Type::TENSOR)
                {
                    if (lhs_type->ty != TypeInfo::Type::TENSOR)
                        return error::compiler(expr.node_info, "Tensor types can only be assigned to other tensors");
                    // Merging tensors is weird.  The forward edges can directly be merged, but thats not always true for the backward edges
                    // This is cause if the tensors are used by multiple nodes, both backward edges will have bound inputs.
                    // In this cases, if the backward edges were merged directly it would cause a runtime error, which is incorrect behaviour
                    // Instead, it should implement the multivariable chain rule where the inputs to the backward edges get summed.
                    // 
                    // The best solution I could come up with to resolve this issue is to check if both the backward edges have their inputs bound
                    // and call the user-overloaded intr __add__ to merge the edges, but this means more special instructions for checking edges...
                    AstExpr lookup_expr;
                    new (&lookup_expr.expr_string) std::string();
                    lookup_expr.ty = ExprType::VAR;
                    lookup_expr.node_info = expr.node_info;
                    lookup_expr.expr_string = "__add__";
                    TypeRef ret;
                    if (codegen_expr_callee_var(scope, lookup_expr, ret))
                        return true;
                    if (ret->ty != TypeInfo::Type::LOOKUP)
                        return error::compiler(expr.node_info, "intr __add__ overload is shadowed in the local scope for tensor merge operation");
                    if (ret->type_lookup.lookup.intrs.size() == 0)
                        return error::compiler(expr.node_info, "Unable to find a intr __add__ overload for tensor merge operation");
                    if (ret->type_lookup.lookup.intrs.size() > 1)
                        return error::compiler(expr.node_info, "intr __add__ was overloaded multiple times.  Merge operation is ambiguous");
                    assert(ret->type_lookup.lookup.intrs.size() == 1);
                    const AstBlock& __add__ = *ret->type_lookup.lookup.intrs[0];
                    if (__add__.signature.vargs.size() < 2)
                        return error::compiler(expr.node_info, "Too few vargs found in intr __add__ signature, expected exactly two");
                    if (__add__.signature.vargs.size() > 2)
                        return error::compiler(expr.node_info, "Too many vargs found in intr __add__ signature, expected exactly two");
                    if (__add__.signature.rets.size() != 1)
                        return error::compiler(expr.node_info, "Expected exactly one return value from intr __add__");
                    std::vector<TypeRef> add_rets;
                    std::string merge_label;
                    std::string end_label;
                    std::unique_ptr<IntrCall> call = std::make_unique<IntrCall>(ret->type_lookup.lookup.ns);
                    return
                        lhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GFwd(expr.node_info)) ||
                        rhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GFwd(expr.node_info)) ||
                        body->add_instruction(instruction::Mrg(expr.node_info)) ||
                        scope.pop(2) ||
                        lhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GBwd(expr.node_info)) ||
                        body->add_instruction(instruction::Einp(expr.node_info)) ||
                        rhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GBwd(expr.node_info)) ||
                        body->add_instruction(instruction::Einp(expr.node_info)) ||
                        body->add_instruction(instruction::LAnd(expr.node_info)) ||
                        body->add_instruction(instruction::Brf(expr.node_info, merge_label)) ||
                        scope.pop(2) ||
                        // multivariable chain rule, sum of the backward paths
                        call->init(__add__) ||
                        call->apply_args(expr.node_info, {}, { lhs_type, rhs_type }) ||
                        call->codegen(scope, add_rets) ||
                        lhs_type->codegen(scope) ||
                        add_rets.front()->codegen(scope) ||
                        body->add_instruction(instruction::SBwd(expr.node_info)) ||
                        scope.pop(2) ||
                        body->add_instruction(instruction::Jmp(expr.node_info, end_label)) ||
                        // Either one or both of the backward edges did not have an input node, a simple mrg is fine
                        body->add_label(expr.node_info, merge_label) ||
                        lhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GBwd(expr.node_info)) ||
                        rhs_type->codegen(scope) ||
                        body->add_instruction(instruction::GBwd(expr.node_info)) ||
                        body->add_instruction(instruction::Mrg(expr.node_info)) ||
                        scope.pop(2) ||
                        body->add_label(expr.node_info, end_label);
                }

                if (rhs_type->ty == TypeInfo::Type::EDGE)
                {
                    if (lhs_type->ty != TypeInfo::Type::EDGE)
                        return error::compiler(expr.node_info, "Edge types can only be assigned to other edges");
                    return
                        lhs_type->codegen(scope) ||
                        rhs_type->codegen(scope) ||
                        body->add_instruction(instruction::Mrg(expr.node_info)) ||
                        scope.pop(2);
                }
                
                return
                    lhs_type->codegen(scope) ||
                    rhs_type->codegen(scope) ||
                    lhs_type->to_obj(expr.node_info, ty) ||
                    ty->codegen(scope) ||
                    body->add_instruction(instruction::Set(expr.node_info)) ||
                    scope.pop(3);
            }
        }

        bool codegen_expr_and(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_or(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        template<bool(TypeInfo::*check)() const, const char* noun, const char* verb, const char* name, class Node>
        bool codegen_expr_binop_bool(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", noun);
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", noun);

            // Checking if both the lhs and rhs are either tensors or edges, in which case
            // generate a new node or block to accomplish the operation
            if (lhs_type->ty == TypeInfo::Type::TENSOR && rhs_type->ty == TypeInfo::Type::TENSOR)
            {
                AstExpr lookup_expr;
                new (&lookup_expr.expr_string) std::string();
                lookup_expr.ty = ExprType::VAR;
                lookup_expr.node_info = expr.node_info;
                lookup_expr.expr_string = name;
                TypeRef ret;
                if (codegen_expr_callee_var(scope, lookup_expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::LOOKUP)
                    return error::compiler(expr.node_info, "def % overload is shadowed in the local scope for tensor % operation", name, noun);
                if (ret->type_lookup.lookup.defs.size() == 0)
                    return error::compiler(expr.node_info, "Unable to find a def % overload for tensor % operation", name, noun);
                if (ret->type_lookup.lookup.defs.size() > 1)
                    return error::compiler(expr.node_info, "def % was overloaded multiple times.  Tensor % operation is ambiguous", name, noun);
                assert(ret->type_lookup.lookup.defs.size() == 1);
                const AstBlock& __def__ = *ret->type_lookup.lookup.defs[0];
                if (__def__.signature.vargs.size() < 2)
                    return error::compiler(expr.node_info, "Too few vargs found in def % signature, expected exactly two", name);
                if (__def__.signature.vargs.size() > 2)
                    return error::compiler(expr.node_info, "Too many vargs found in def % signature, expected exactly two", name);
                if (__def__.signature.rets.size() != 1)
                    return error::compiler(expr.node_info, "Expected exactly one return value from def %", name);
                std::unique_ptr<DefCall> call = std::make_unique<DefCall>(ret->type_lookup.lookup.ns);
                return
                    call->init(__def__) ||
                    call->apply_args(expr.node_info, {}, { lhs_type, rhs_type }) ||
                    call->codegen(scope, rets);
            }

            if (lhs_type->ty == TypeInfo::Type::EDGE && rhs_type->ty == TypeInfo::Type::EDGE)
            {
                AstExpr lookup_expr;
                new (&lookup_expr.expr_string) std::string();
                lookup_expr.ty = ExprType::VAR;
                lookup_expr.node_info = expr.node_info;
                lookup_expr.expr_string = name;
                TypeRef ret;
                if (codegen_expr_callee_var(scope, lookup_expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::LOOKUP)
                    return error::compiler(expr.node_info, "intr % overload is shadowed in the local scope for edge % operation", name, noun);
                if (ret->type_lookup.lookup.intrs.size() == 0)
                    return error::compiler(expr.node_info, "Unable to find a intr % overload for edge % operation", name, noun);
                if (ret->type_lookup.lookup.intrs.size() > 1)
                    return error::compiler(expr.node_info, "intr % was overloaded multiple times.  Edge % operation is ambiguous", name, noun);
                assert(ret->type_lookup.lookup.intrs.size() == 1);
                const AstBlock& __intr__ = *ret->type_lookup.lookup.intrs[0];
                if (__intr__.signature.vargs.size() < 2)
                    return error::compiler(expr.node_info, "Too few vargs found in intr % signature, expected exactly two", name);
                if (__intr__.signature.vargs.size() > 2)
                    return error::compiler(expr.node_info, "Too many vargs found in intr % signature, expected exactly two", name);
                if (__intr__.signature.rets.size() != 1)
                    return error::compiler(expr.node_info, "Expected exactly one return value from intr %", name);
                std::unique_ptr<IntrCall> call = std::make_unique<IntrCall>(ret->type_lookup.lookup.ns);
                return
                    call->init(__intr__) ||
                    call->apply_args(expr.node_info, {}, { lhs_type, rhs_type }) ||
                    call->codegen(scope, rets);
            }

            if (!((*lhs_type).*check)())
                return error::compiler(expr.node_info, "Unable to % values with type '%'", verb, lhs_type->to_string());
            if (lhs_type != rhs_type)  // Only exact matches are allowed.  Explicit casting is otherwise nessisary
                return error::compiler(expr.node_info, "Unable to % a value of type '%' with a value of type '%'",
                    verb, lhs_type->to_string(), rhs_type->to_string());

            TypeRef type;
            if (lhs_type->to_obj(expr.node_info, type))
                return true;
            TypeRef ret = type_manager->create_bool(
                TypeInfo::Category::CONST,
                [type, &expr, lhs_type, rhs_type](Scope& scope) -> bool {
                    return
                        lhs_type->codegen(scope) ||  // codegen the lhs
                        rhs_type->codegen(scope) ||  // codegen the rhs
                        type->codegen(scope) ||
                        body->add_instruction(Node(expr.node_info)) ||
                        scope.pop(2);  // pops ty, rhs, lhs; pushes ret
                });
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        char eq_noun[] = "equality"                ; char eq_verb[] = "compare"; char eq_name[] = "__eq__";
        char ne_noun[] = "inequality"              ; char ne_verb[] = "compare"; char ne_name[] = "__ne__";
        char gt_noun[] = "greater than"            ; char gt_verb[] = "compare"; char gt_name[] = "__gt__";
        char lt_noun[] = "less than"               ; char lt_verb[] = "compare"; char lt_name[] = "__lt__";
        char ge_noun[] = "greater than or equal to"; char ge_verb[] = "compare"; char ge_name[] = "__ge__";
        char le_noun[] = "less than or equal to"   ; char le_verb[] = "compare"; char le_name[] = "__le__";
        op codegen_expr_eq = codegen_expr_binop_bool<&TypeInfo::check_eq, eq_noun, eq_verb, eq_name, instruction::Eq>;
        op codegen_expr_ne = codegen_expr_binop_bool<&TypeInfo::check_ne, ne_noun, ne_verb, ne_name, instruction::Ne>;
        op codegen_expr_gt = codegen_expr_binop_bool<&TypeInfo::check_gt, gt_noun, gt_verb, gt_name, instruction::Gt>;
        op codegen_expr_lt = codegen_expr_binop_bool<&TypeInfo::check_lt, lt_noun, lt_verb, lt_name, instruction::Lt>;
        op codegen_expr_ge = codegen_expr_binop_bool<&TypeInfo::check_ge, ge_noun, ge_verb, ge_name, instruction::Ge>;
        op codegen_expr_le = codegen_expr_binop_bool<&TypeInfo::check_le, le_noun, le_verb, le_name, instruction::Le>;

        bool codegen_expr_cast(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the cast operation");
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the cast operation");

            // TODO: implement ellipses in reshaping
            if (lhs_type->ty == TypeInfo::Type::DLTYPE)
            {
                // Its a reshape operation
                AstExpr lookup_expr;
                new (&lookup_expr.expr_string) std::string();
                lookup_expr.ty = ExprType::VAR;
                lookup_expr.node_info = expr.node_info;
                lookup_expr.expr_string = "__cast__";
                TypeRef ret;
                if (codegen_expr_callee_var(scope, lookup_expr, ret))
                    return true;

                if (rhs_type->ty == TypeInfo::Type::TENSOR)
                {
                    if (ret->ty != TypeInfo::Type::LOOKUP)
                        return error::compiler(expr.node_info, "def __cast__ overload is shadowed in the local scope for tensor cast operation");
                    if (ret->type_lookup.lookup.defs.size() == 0)
                        return error::compiler(expr.node_info, "Unable to find a def __cast__ overload for tensor cast operation");
                    if (ret->type_lookup.lookup.defs.size() > 1)
                        return error::compiler(expr.node_info, "def __cast__ was overloaded multiple times.  Tensor cast operation is ambiguous");
                    assert(ret->type_lookup.lookup.defs.size() == 1);
                    const AstBlock& __cast__ = *ret->type_lookup.lookup.defs[0];
                    if (__cast__.signature.vargs.size() < 1)
                        return error::compiler(expr.node_info, "Too few vargs found in def __cast__ signature, expected exactly one");
                    if (__cast__.signature.vargs.size() > 1)
                        return error::compiler(expr.node_info, "Too many vargs found in def __cast__ signature, expected exactly one");
                    if (__cast__.signature.rets.size() != 1)
                        return error::compiler(expr.node_info, "Expected exactly one return value from def __cast__");
                    std::unique_ptr<DefCall> call = std::make_unique<DefCall>(ret->type_lookup.lookup.ns);
                    return
                        call->init(__cast__) ||
                        call->apply_args(expr.node_info, {
                                { "out_fp", lhs_type->type_dltype.fp },
                                { "out_shape", lhs_type->type_dltype.shape },
                            }, { rhs_type }) ||
                        call->codegen(scope, rets);
                }
                if (rhs_type->ty == TypeInfo::Type::EDGE)
                {
                    if (ret->ty != TypeInfo::Type::LOOKUP)
                        return error::compiler(expr.node_info, "intr __cast__ overload is shadowed in the local scope for edge cast operation");
                    if (ret->type_lookup.lookup.intrs.size() == 0)
                        return error::compiler(expr.node_info, "Unable to find a intr __cast__ overload for edge cast operation");
                    if (ret->type_lookup.lookup.intrs.size() > 1)
                        return error::compiler(expr.node_info, "intr __cast__ was overloaded multiple times.  Edge cast operation is ambiguous");
                    assert(ret->type_lookup.lookup.intrs.size() == 1);
                    const AstBlock& __cast__ = *ret->type_lookup.lookup.intrs[0];
                    if (__cast__.signature.vargs.size() < 1)
                        return error::compiler(expr.node_info, "Too few vargs found in intr __cast__ signature, expected exactly two");
                    if (__cast__.signature.vargs.size() > 1)
                        return error::compiler(expr.node_info, "Too many vargs found in intr __cast__ signature, expected exactly two");
                    if (__cast__.signature.rets.size() != 1)
                        return error::compiler(expr.node_info, "Expected exactly one return value from intr __cast__");
                    std::unique_ptr<IntrCall> call = std::make_unique<IntrCall>(ret->type_lookup.lookup.ns);
                    return
                        call->init(__cast__) ||
                        call->apply_args(expr.node_info, {
                                { "out_fp", lhs_type->type_dltype.fp },
                                { "out_shape", lhs_type->type_dltype.shape },
                            }, { rhs_type }) ||
                        call->codegen(scope, rets);
                }
                return error::compiler(expr.expr_binary.right->node_info, "Expected either a tensor or an edge in a reshape cast operation");
            }

            // Normal cast
            if (lhs_type->ty != TypeInfo::Type::TYPE)
                return error::compiler(expr.expr_binary.left->node_info, "Expected a type as the left side of a cast operation");

            TypeRef type, ret;
            if (rhs_type->to_obj(expr.node_info, type))
                return true;

            switch (lhs_type->type_type.base->ty)
            {
            case TypeInfo::Type::INT:
                if (!rhs_type->check_xint())
                    return error::compiler(expr.node_info, "Unable to cast type '%' to int", rhs_type->to_string());
                ret = type_manager->create_int(
                    TypeInfo::Category::CONST,
                    [&expr, rhs_type, type](Scope& scope) -> bool {
                        return
                            rhs_type->codegen(scope) ||
                            type->codegen(scope) ||
                            body->add_instruction(instruction::XInt(expr.node_info)) ||
                            scope.pop();
                    });
                break;
            case TypeInfo::Type::FLOAT:
                if (!rhs_type->check_xflt())
                    return error::compiler(expr.node_info, "Unable to cast type '%' to float", rhs_type->to_string());
                ret = type_manager->create_float(
                    TypeInfo::Category::CONST,
                    [&expr, rhs_type, type](Scope& scope) -> bool {
                        return
                            rhs_type->codegen(scope) ||
                            type->codegen(scope) ||
                            body->add_instruction(instruction::XFlt(expr.node_info)) ||
                            scope.pop();
                    });
                break;
            case TypeInfo::Type::STR:
                if (!rhs_type->check_xstr())
                    return error::compiler(expr.node_info, "Unable to cast type '%' to string", rhs_type->to_string());
                ret = type_manager->create_string(
                    TypeInfo::Category::CONST,
                    [&expr, rhs_type, type](Scope& scope) -> bool {
                        return
                            rhs_type->codegen(scope) ||
                            type->codegen(scope) ||
                            body->add_instruction(instruction::XStr(expr.node_info)) ||
                            scope.pop();
                    });
                break;
            default:
                return error::compiler(expr.expr_binary.left->node_info, "Casting to type '%' is not supported", lhs_type->type_type.base->to_string());
            }
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_idx(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef callee;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr.expr_index.expr, callee))
                return true;
            if (callee->ty == TypeInfo::Type::TENSOR)
                return error::compiler(expr.node_info, "Tensor slicing has not been implemented");
            if (callee->ty == TypeInfo::Type::EDGE)
                return error::compiler(expr.node_info, "Edge slicing has not been implemented");
            
            if (expr.expr_index.args.size() != 1)
                return error::compiler(expr.node_info, "Expected a single index argument, recieved %", expr.expr_index.args.size());
            const AstExprIndex::Elem& expr_arg = expr.expr_index.args.front();
            assert(expr_arg.ty != AstExprIndex::Elem::Type::INVALID);
            if (expr_arg.ty == AstExprIndex::Elem::Type::ELLIPSES)
                return error::compiler(expr.node_info, "Invalid usage of ellipses (...)");
            if (expr_arg.ty == AstExprIndex::Elem::Type::SLICE)
            {
                // Do a slice
                return error::compiler(expr.node_info, "Internal error: slicing has not been implemented");
            }
            if (expr_arg.ty == AstExprIndex::Elem::Type::DIRECT)
            {
                // Do an index
                TypeRef ret;
                TypeRef callee_type;
                if (callee->to_obj(expr.node_info, callee_type))
                    return true;
                TypeRef arg;
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *expr_arg.lhs, arg))
                    return true;
                if (arg->ty != TypeInfo::Type::INT)
                    return error::compiler(expr_arg.lhs->node_info, "Expected integer as index argument, recieved %", arg);
                switch (callee->ty)
                {
                case TypeInfo::Type::ARRAY:
                    ret = type_manager->duplicate(
                        callee->cat,
                        [&expr, callee, arg, callee_type](Scope& scope) -> bool {
                            return
                                callee->codegen(scope) ||
                                arg->codegen(scope) ||
                                callee_type->codegen(scope) ||
                                body->add_instruction(instruction::Idx(expr.node_info)) ||
                                scope.pop(2);
                        }, callee->type_array.elem);
                    break;
                case TypeInfo::Type::TUPLE:
                    if (expr_arg.lhs->ty != ExprType::LIT_INT)
                        return error::compiler(expr_arg.lhs->node_info, "Tuple indexing arguments must be known at compile time");
                    if (expr_arg.lhs->expr_int < 0 || callee->type_tuple.elems.size() <= expr_arg.lhs->expr_int)
                        return error::compiler(expr_arg.lhs->node_info, "Tuple index % is out of range", expr_arg.lhs->expr_int);
                    ret = type_manager->duplicate(
                        callee->cat,
                        [&expr, callee, arg, callee_type](Scope& scope) -> bool {
                            return
                                callee->codegen(scope) ||
                                arg->codegen(scope) ||  // This should compile into a new instruction
                                callee_type->codegen(scope) ||
                                body->add_instruction(instruction::Idx(expr.node_info)) ||
                                scope.pop(2);
                        }, callee->type_tuple.elems[expr_arg.lhs->expr_int]);
                default:
                    return error::compiler(expr.node_info, "Unable to index into type %", callee->to_string());
                }
                if (!ret)
                    return true;
                rets.push_back(ret);
                return false;
            }
            return error::compiler(expr.node_info, "Internal error: invalid index elem enum %", (int)expr_arg.ty);
        }

        bool codegen_expr_dot(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef lhs;
            if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, *expr.expr_name.expr, lhs))
                return true;
            switch (lhs->ty)
            {
            case TypeInfo::Type::LOOKUP:
            {
                const CodeModule::LookupResult& lookup = lhs->type_lookup.lookup;
                // A dot operator on a lookup operation needs to access a namespace
                CodeModule::LookupResult result;
                for (const CodeModule::Node* node : lookup.nodes)
                {
                    if (node->attrs.contains(expr.expr_name.val))
                    {
                        for (const auto& attr : node->attrs.at(expr.expr_name.val))
                        {
                            switch (attr.index())
                            {
                            case CodeModule::AttrType::NODE:
                                result.nodes.push_back(&std::get<CodeModule::Node>(attr));
                                break;
                            case CodeModule::AttrType::STRUCT:
                                result.structs.push_back(std::get<const AstStruct*>(attr));
                                break;
                            case CodeModule::AttrType::FUNC:
                                result.fns.push_back(std::get<const AstFn*>(attr));
                                break;
                            case CodeModule::AttrType::DEF:
                                result.defs.push_back(std::get<const AstBlock*>(attr));
                                break;
                            default:
                                assert(false);
                            }
                        }
                    }
                    if (node->intrs.contains(expr.expr_name.val))
                        result.intrs = node->intrs.at(expr.expr_name.val);
                    if (node->inits.contains(expr.expr_name.val))
                        result.inits = node->inits.at(expr.expr_name.val);
                }
                TypeRef type = type_manager->create_lookup(expr.expr_name.val, std::move(result));
                if (!type)
                    return true;
                rets.push_back(type);
                return false;
            }
            case TypeInfo::Type::STRUCT:
                return error::compiler(expr.node_info, "struct member access has not been implemented");
            default:
                return error::compiler(expr.node_info, "Unable to perform a member access operation on the given type");
            }
            
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_decl(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef decl_type;
            if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, *expr.expr_name.expr, decl_type))
                return true;
            if (decl_type->ty != TypeInfo::Type::TYPE)
            {
                if (decl_type->ty != TypeInfo::Type::DLTYPE)
                    return error::compiler(expr.node_info, "Variable declarations require a type");
                
                // Its either a tensor declaration or edge declaration depending on the context
                TypeRef ret;
                switch (body_type)
                {
                case BodyType::INVALID:
                    return error::compiler(expr.node_info, "Internal error: body_type enum is invalid");
                case BodyType::STRUCT:
                case BodyType::DEF:
                case BodyType::FN:
                    // In all of these contexts, its a tensor
                    if (decl_type->type_dltype.tensor->codegen(scope))
                        return true;
                    ret = type_manager->create_tensor(
                        TypeInfo::Category::DEFAULT,
                        [&expr](Scope& scope) -> bool {
                            Scope::StackVar var;
                            return
                                scope.at(expr.expr_name.val, var) ||
                                body->add_instruction(instruction::Dup(expr.node_info, var.ptr)) ||
                                scope.push();
                        });
                    break;
                case BodyType::INTR:
                    // Only in intrs are tensor declarations actually edges
                    if (decl_type->type_dltype.edge->codegen(scope))
                        return true;
                    ret = type_manager->create_edge(
                        TypeInfo::Category::DEFAULT,
                        [&expr](Scope& scope) -> bool {
                            Scope::StackVar var;
                            return
                                scope.at(expr.expr_name.val, var) ||
                                body->add_instruction(instruction::Dup(expr.node_info, var.ptr)) ||
                                scope.push();
                        });
                    break;
                default:
                    return error::compiler(expr.node_info, "Internal error: body_type enum is out of range");
                }
                if (!ret || scope.add(expr.expr_name.val, ret, expr.node_info))
                    return true;
                rets.push_back(ret);
                return false;
            }
            
            TypeRef type = decl_type->type_type.base;
            if (type->cat != TypeInfo::Category::DEFAULT)
                return error::compiler(expr.node_info, "Invalid type category for variable declaration");
            if (type->codegen(scope))
                return true;
            type->codegen = [&expr](Scope& scope) {
                Scope::StackVar var;
                if (scope.at(expr.expr_name.val, var))
                    return error::compiler(expr.node_info, "Undefined variable %", expr.expr_name.val);
                return
                    body->add_instruction(instruction::Dup(expr.node_info, var.ptr)) ||
                    scope.push();
            };
            if (scope.add(expr.expr_name.val, type, expr.node_info))
                return true;
            rets.push_back(type);
            return false;
        }

        bool codegen_expr_cargs(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            // Handling special cases
            if (expr.expr_call.callee->ty == ExprType::KW)
            {
                switch (expr.expr_call.callee->expr_kw)
                {
                case ExprKW::TYPE:
                {
                    if (expr.expr_call.args.size() != 1)
                        return error::compiler(expr.node_info, "type requires exactly 1 carg");
                    TypeRef arg_type;
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr.expr_call.args[0], arg_type))
                        return true;
                    TypeRef type = type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, arg_type);
                    if (!type)
                        return true;
                    rets.push_back(type);
                    return false;
                }
                case ExprKW::INIT:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'init'");
                case ExprKW::FTY:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'fty'");
                case ExprKW::BOOL:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'bool'");
                case ExprKW::INT:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'int'");
                case ExprKW::FLOAT:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'float'");
                case ExprKW::STR:
                    return error::compiler(expr.node_info, "Invalid use of keyword 'str'");
                case ExprKW::ARRAY:
                {
                    if (expr.expr_call.args.size() != 1)
                        return error::compiler(expr.node_info, "Array type requires exactly 1 carg");
                    TypeRef arg_type;
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr.expr_call.args[0], arg_type))
                        return true;
                    if (arg_type->ty != TypeInfo::Type::TYPE)
                        return error::compiler(expr.expr_call.args[0].node_info, "The carg of array must be a type, recieved %", arg_type->to_string());
                    TypeRef tmp = type_manager->create_array(TypeInfo::Category::VIRTUAL, nullptr, arg_type->type_type.base);
                    if (!tmp)
                        return true;
                    TypeRef type;
                    if (tmp->to_obj(expr.node_info, type))
                        return true;
                    rets.push_back(type);
                    return false;
                }
                case ExprKW::TUPLE:
                {
                    std::vector<TypeRef> raw_arg_types;
                    for (const auto& arg : expr.expr_call.args)
                        if (codegen_expr(scope, arg, raw_arg_types))
                            return true;
                    std::vector<TypeRef> arg_types;
                    for (const auto& raw_arg_type : raw_arg_types)
                    {
                        if (raw_arg_type->ty != TypeInfo::Type::TYPE)
                            return error::compiler(expr.node_info, "The cargs of a tuple must be types, recieved %", raw_arg_type->to_string());
                        arg_types.push_back(raw_arg_type->type_type.base);
                    }
                    TypeRef tmp = type_manager->create_tuple(TypeInfo::Category::VIRTUAL, nullptr, arg_types);
                    if (!tmp)
                        return true;
                    TypeRef type;
                    if (tmp->to_obj(expr.node_info, type))
                        return true;
                    rets.push_back(type);
                    return false;
                }
                case ExprKW::F16:
                case ExprKW::F32:
                case ExprKW::F64:
                    // Let the codegen fall through, it'll get caught by codegen_expr_kw and handled in the code below
                    break;
                default:
                    return error::compiler(expr.expr_call.callee->node_info, "Internal error: invalid keyword enum");
                }
            }

            // In general, codegen the callee, and see what happens
            TypeRef callee;
            if (codegen_expr_callee(scope, *expr.expr_call.callee, callee))
                return true;
            // There might be other types that can get cargs, but I just don't remember them at the moment
            if (callee->ty == TypeInfo::Type::LOOKUP)
            {
                // This function is only called when there are not vargs applied to the result.
                // Otherwise codegen_expr_callee_cargs is called.
                // So in this call, when cargs are applied to a lookup, the lookup can be disabmiguated
                // to either an init or struct call directly.
                // First try to match the args against the inits
                std::vector<std::pair<const AstInit*, std::map<std::string, TypeRef>>> init_matches;
                for (const AstInit* init_elem : callee->type_lookup.lookup.inits)
                {
                    std::map<std::string, TypeRef> cargs;
                    Scope sig_scope{ nullptr };
                    if (!match_carg_sig(scope, sig_scope, init_elem->signature.cargs, expr.expr_call.args, cargs))
                        init_matches.push_back({ init_elem, std::move(cargs) });
                }
                if (init_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to init '%' is ambiguous", callee->type_lookup.name);
                if (init_matches.size() == 1)
                {
                    // Perfect match, return it
                    std::unique_ptr<InitCall> call = std::make_unique<InitCall>(callee->type_lookup.lookup.ns);
                    return
                        call->init(*init_matches[0].first) ||
                        call->apply_args(expr.node_info, init_matches[0].second) ||
                        call->codegen(scope, rets);
                }
                // Non of the inits matched the cargs, so see if a struct matches
                std::vector<std::pair<const AstStruct*, std::map<std::string, TypeRef>>> struct_matches;
                for (const AstStruct* struct_elem : callee->type_lookup.lookup.structs)
                {
                    std::map<std::string, TypeRef> cargs;
                    Scope sig_scope{ nullptr };
                    if (!match_carg_sig(scope, sig_scope, struct_elem->signature.cargs, expr.expr_call.args, cargs))
                        struct_matches.push_back({ struct_elem, std::move(cargs) });
                }
                if (init_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to struct '%' is ambiguous", callee->type_lookup.name);
                if (init_matches.size() == 1)
                {
                    // Perfect match, return it
                    return error::compiler(expr.node_info, "Interal error: not implemented");
                }
                // Couldn't find anything, its a user error
                return error::compiler(expr.node_info, "Unable to find an init or struct '%' with the given cargs", callee->type_lookup.name);
            }
            else if (callee->ty == TypeInfo::Type::FTY)
            {
                // Its a tensor declaration
                std::vector<TypeRef> args;
                for (const auto& expr_arg : expr.expr_call.args)
                    if (codegen_expr(scope, expr_arg, args))
                        return true;
                // Confirming that all the arguments are valid
                for (const auto& arg : args)
                {
                    if (arg->ty == TypeInfo::Type::INT)
                    {
                        if (arg->cat == TypeInfo::Category::VIRTUAL)
                            return error::compiler(expr.node_info, "Arguments to a tensor declaration must be concrete integers");
                    }
                    else if (arg->ty == TypeInfo::Type::UNPACK)
                    {
                        if (arg->type_array.elem->ty != TypeInfo::Type::INT)
                            return error::compiler(expr.node_info, "Arguments to a tensor declaration must be concrete integers");
                    }
                    else
                        return error::compiler(expr.node_info, "Arguments to a tensor declaration must be concrete integers");
                }
                // Provide codegens for everything.  Then whatever uses the result
                // can do whatever it wants to with decent efficiency
                TypeRef tensor = type_manager->create_tensor(
                    TypeInfo::Category::DEFAULT,
                    [&expr, callee, args](Scope& scope) -> bool {
                        // Setup for the tensor generation
                        // codegen(fp)
                        if (callee->codegen(scope))
                            return true;
                        // for arg in args: codegen(arg)
                        for (const auto& arg : args)
                            if (arg->codegen(scope))
                                return true;
                        // new int <start_val>
                        size_t start_val = 0;
                        for (const auto& arg : args)
                            if (arg->ty == TypeInfo::Type::INT)
                                start_val++;
                        size_t addr, int_addr;
                        if (bc->add_obj_int(addr, start_val) ||
                            bc->add_type_int(int_addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                            body->add_instruction(instruction::Cpy(expr.node_info))
                            ) return true;
                        // for arg in args: dup n; dup arg; len; new type int; iadd
                        for (size_t i = 0; i < args.size(); i++)
                            if (args[i]->ty == TypeInfo::Type::UNPACK)
                            {
                                if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||
                                    body->add_instruction(instruction::Dup(expr.node_info, args.size() - i + 1)) ||
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                    body->add_instruction(instruction::Arr(expr.node_info)) ||
                                    body->add_instruction(instruction::Len(expr.node_info)) ||
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                    body->add_instruction(instruction::IAdd(expr.node_info))
                                    ) return true;
                            }
                        // tsr
                        if (body->add_instruction(instruction::Tsr(expr.node_info)))
                            return true;
                        // Setup done.  Current stack: fp, *args, n, tensor
                        
                        // ready to create the different ints needed
                        size_t int0, int1;
                        if (bc->add_obj_int(int0, 0) ||
                            bc->add_obj_int(int1, 1)
                            ) return true;
                        // n=0 -> forward edge, n=1 -> backward edge
                        for (size_t n = 0; n < 2; n++)
                        {
                            // Setup for edge generation
                            if (body->add_instruction(instruction::Dup(expr.node_info, args.size() + 2)))  // dup fp
                                return true;
                            for (size_t i = 0; i < args.size(); i++)  // for arg in args: dup <arg>
                                if (body->add_instruction(instruction::Dup(expr.node_info, args.size() + 2)))
                                    return true;
                            if (body->add_instruction(instruction::Dup(expr.node_info, args.size() + 2)))  // dup n
                                return true;

                            // Expanding each of the packed arguments onto the stack
                            for (size_t k = 0; k < args.size(); k++)
                                // TypeInfo::Type::INT will get handled automatically due to the setup
                                if (args[k]->ty == TypeInfo::Type::UNPACK)
                                {
                                    std::string loop_label = label_prefix(expr.node_info) + "_loop" + std::to_string(n) + "_" + std::to_string(k);
                                    std::string end_label = label_prefix(expr.node_info) + "_end" + std::to_string(n) + "_" + std::to_string(k);

                                    // Break condition for the loop over args[k]
                                    if (body->add_instruction(instruction::New(expr.node_info, int0)) ||                 // new int 0  # i
                                        body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                        body->add_instruction(instruction::Cpy(expr.node_info)) ||                       // cpy
                                        body->add_label(expr.node_info, loop_label) ||                                   // :loop
                                        body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 1)) ||  // dup <args[k]>
                                        body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||                       // arr
                                        body->add_instruction(instruction::Len(expr.node_info)) ||                       // len
                                        body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                    // dup i
                                        body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                        body->add_instruction(instruction::Eq(expr.node_info)) ||                        // eq
                                        body->add_instruction(instruction::Brt(expr.node_info, end_label))               // brt end
                                        ) return true;

                                    // Placing the next element from args[k] onto the stack
                                    if (body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 1)) ||  // dup args[k]
                                        body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                    // dup i
                                        body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||                       // arr
                                        body->add_instruction(instruction::Idx(expr.node_info))                          // idx
                                        ) return true;

                                    // Reorganizing the stack
                                    // 1 for n, "args.size() - k" args, and 1 for i
                                    for (int j = 0; j < args.size() - k + 2; j++)
                                    {
                                        if (body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 2)) ||
                                            body->add_instruction(instruction::Pop(expr.node_info, args.size() - k + 3))
                                            ) return true;
                                    }

                                    // Incrementing i
                                    if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||         // dup i
                                        body->add_instruction(instruction::New(expr.node_info, int1)) ||      // new int 1
                                        body->add_instruction(instruction::New(expr.node_info, int_addr)) ||  // new type int
                                        body->add_instruction(instruction::IAdd(expr.node_info))              // iadd
                                        ) return true;

                                    // Looping back and stack cleanup
                                    if (body->add_instruction(instruction::Jmp(expr.node_info, loop_label)) ||           // jmp loop
                                        body->add_label(expr.node_info, end_label) ||                                    // :end
                                        body->add_instruction(instruction::Pop(expr.node_info, args.size() - k + 1)) ||  // pop args[k]
                                        body->add_instruction(instruction::Pop(expr.node_info, 0))                       // pop i
                                        ) return true;
                                }
                            
                            // Generating the edge and binding it to the tensor
                            if (body->add_instruction(instruction::Edg(expr.node_info)))
                                return true;
                            if (n == 0)
                            {
                                if (body->add_instruction(instruction::SFwd(expr.node_info)))
                                    return true;
                            }
                            else
                            {
                                if (body->add_instruction(instruction::SBwd(expr.node_info)))
                                    return true;
                            }
                        }

                        // Cleaning up the stack from the initial setup
                        for (size_t n = 0; n < args.size() + 2; n++)
                            if (body->add_instruction(instruction::Pop(expr.node_info, 1)))
                                return true;
                        return scope.pop(args.size());
                    });
                if (!tensor)
                    return true;
                TypeRef edge = type_manager->create_edge(
                    TypeInfo::Category::DEFAULT,
                    [&expr, callee, args](Scope& scope) -> bool {
                        // codegen(fp)
                        if (callee->codegen(scope))
                            return true;
                        // for arg in args: codegen(arg)
                        for (const auto& arg : args)
                            if (arg->codegen(scope))
                                return true;
                        // new int <start_val>
                        size_t start_val = 0;
                        for (const auto& arg : args)
                            if (arg->ty == TypeInfo::Type::INT)
                                start_val++;
                        size_t addr, int_addr;
                        if (bc->add_obj_int(addr, start_val) ||
                            bc->add_type_int(int_addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                            body->add_instruction(instruction::Cpy(expr.node_info))
                            ) return true;
                        // for arg in args: dup n; dup arg; len; new type int; iadd
                        for (size_t i = 0; i < args.size(); i++)
                            if (args[i]->ty == TypeInfo::Type::UNPACK)
                            {
                                if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||
                                    body->add_instruction(instruction::Dup(expr.node_info, args.size() - i + 1)) ||
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                    body->add_instruction(instruction::Arr(expr.node_info)) ||
                                    body->add_instruction(instruction::Len(expr.node_info)) ||
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                    body->add_instruction(instruction::IAdd(expr.node_info))
                                    ) return true;
                            }
                        // Current stack: fp, *args, n

                        size_t zero_addr, one_addr;
                        if (bc->add_obj_int(zero_addr, 0) ||
                            bc->add_obj_int(one_addr, 1)
                            ) return true;
                        // Expanding each of the packed arguments onto the stack
                        for (size_t k = 0; k < args.size(); k++)
                            // TypeInfo::Type::INT will get handled automatically due to the setup
                            if (args[k]->ty == TypeInfo::Type::UNPACK)
                            {
                                // Stack during the loop: fp, expanded values..., *args, n, i
                                std::string loop_label = label_prefix(expr.node_info) + "_loop_" + std::to_string(k);
                                std::string end_label = label_prefix(expr.node_info) + "_end_" + std::to_string(k);

                                // Break condition for the loop over args[k]
                                if (body->add_instruction(instruction::New(expr.node_info, zero_addr)) ||            // new int 0  # i
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                    body->add_instruction(instruction::Cpy(expr.node_info)) ||                       // cpy
                                    body->add_label(expr.node_info, loop_label) ||                                   // :loop
                                    body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 1)) ||  // dup <args[k]>
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                    body->add_instruction(instruction::Arr(expr.node_info)) ||                       // arr
                                    body->add_instruction(instruction::Len(expr.node_info)) ||                       // len
                                    body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                    // dup i
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                    body->add_instruction(instruction::Eq(expr.node_info)) ||                        // eq
                                    body->add_instruction(instruction::Brt(expr.node_info, end_label))               // brt end
                                    ) return true;

                                // Placing the next element from args[k] onto the stack
                                if (body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 1)) ||  // dup args[k]
                                    body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                    // dup i
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||             // new type int
                                    body->add_instruction(instruction::Arr(expr.node_info)) ||                       // arr
                                    body->add_instruction(instruction::Idx(expr.node_info))                          // idx
                                    ) return true;

                                // Reorganizing the stack
                                // 1 for n, "args.size() - k" args, and 1 for i
                                for (int j = 0; j < args.size() - k + 2; j++)
                                {
                                    if (body->add_instruction(instruction::Dup(expr.node_info, args.size() - k + 2)) ||
                                        body->add_instruction(instruction::Pop(expr.node_info, args.size() - k + 3))
                                        ) return true;
                                }

                                // Incrementing i
                                if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||         // dup i
                                    body->add_instruction(instruction::New(expr.node_info, one_addr)) ||  // new int 1
                                    body->add_instruction(instruction::New(expr.node_info, int_addr)) ||  // new type int
                                    body->add_instruction(instruction::IAdd(expr.node_info))              // iadd
                                    ) return true;

                                // Looping back and stack cleanup
                                if (body->add_instruction(instruction::Jmp(expr.node_info, loop_label)) ||           // jmp loop
                                    body->add_label(expr.node_info, end_label) ||                                    // :end
                                    body->add_instruction(instruction::Pop(expr.node_info, args.size() - k + 1)) ||  // pop args[k]
                                    body->add_instruction(instruction::Pop(expr.node_info, 0))                       // pop i
                                    ) return true;
                            }

                        // Generating the edge
                        return
                            scope.pop(args.size()) ||
                            body->add_instruction(instruction::Edg(expr.node_info));
                    });
                if (!edge)
                    return true;
                TypeRef tmp = type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp)
                    return true;
                TypeRef shape = type_manager->create_array(
                    TypeInfo::Category::CONST,
                    [&expr, args](Scope& scope) -> bool {
                        size_t int_addr;
                        if (bc->add_type_int(int_addr))
                            return true;
                        bool init = false;
                        int stack_size = 0;
                        for (TypeRef arg : args)
                        {
                            if (arg->ty == TypeInfo::Type::UNPACK)
                            {
                                if (stack_size)
                                {
                                    if (body->add_instruction(instruction::Agg(expr.node_info, stack_size)) ||
                                        scope.pop((size_t)stack_size - 1)
                                        ) return true;
                                    if (init)
                                    {
                                        if (body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                            body->add_instruction(instruction::Arr(expr.node_info)) ||
                                            body->add_instruction(instruction::Add(expr.node_info)) ||
                                            scope.pop()
                                            ) return true;
                                    }
                                    stack_size = 0;
                                    init = true;
                                }
                                if (arg->codegen(scope))
                                    return true;
                                if (init)
                                {
                                    if (body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||
                                        body->add_instruction(instruction::Add(expr.node_info)) ||
                                        scope.pop()
                                        ) return true;
                                }
                            }
                            else
                            {
                                if (arg->codegen(scope))
                                    return true;
                                stack_size++;
                            }
                        }

                        if (stack_size)
                        {
                            if (body->add_instruction(instruction::Agg(expr.node_info, stack_size)) ||
                                scope.pop(stack_size - 1)
                                ) return true;
                        }
                        if (init)
                        {
                            if (body->add_instruction(instruction::New(expr.node_info, int_addr)) ||
                                body->add_instruction(instruction::Arr(expr.node_info)) ||
                                body->add_instruction(instruction::Add(expr.node_info)) ||
                                scope.pop()
                                ) return true;
                        }
                        return false;
                    }, tmp);
                if (!shape)
                    return true;
                TypeRef fp = type_manager->duplicate(callee);
                if (!fp)
                    return true;
                TypeRef type = type_manager->create_dltype(tensor, edge, shape, fp);
                if (!type)
                    return true;
                rets.push_back(type);
                return false;
            }
            else
                return error::compiler(expr.node_info, "cargs cannot be applied to the given type");
        }

        bool codegen_expr_vargs(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            // Check if its a len() call
            if (expr.expr_call.callee->ty == ExprType::KW && expr.expr_call.callee->expr_kw == ExprKW::LEN)
            {
                // its a length call
                if (expr.expr_call.args.size() != 1)
                    return error::compiler(expr.node_info, "len call requires exactly one argument");
                TypeRef arg;
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr.expr_call.args.front(), arg))
                    return true;
                if (!arg->check_len())
                    return error::compiler(expr.node_info, "Unable to get the length of type %", arg->to_string());
                TypeRef type;
                if (arg->to_obj(expr.node_info, type))
                    return true;
                TypeRef ret = type_manager->create_int(
                    TypeInfo::Category::CONST,
                    [&expr, type, arg](Scope& scope) -> bool {
                        return
                            arg->codegen(scope) ||
                            type->codegen(scope) ||
                            body->add_instruction(instruction::Len(expr.node_info)) ||
                            scope.pop();
                    });
                if (!ret)
                    return true;
                rets.push_back(ret);
                return false;
            }

            TypeRef callee;
            if (codegen_expr_callee(scope, *expr.expr_call.callee, callee))
                return true;
            // codegening each of the vargs
            std::vector<TypeRef> param_vargs;
            for (const AstExpr& arg : expr.expr_call.args)
                if (codegen_expr_multi_ret<TypeInfo::NonVirtual>(scope, arg, param_vargs))
                    return true;
            bool check_def = true;
            bool check_intr = true;
            for (TypeRef varg : param_vargs)
            {
                if (varg->ty != TypeInfo::Type::TENSOR)
                    check_def = false;
                if (varg->ty != TypeInfo::Type::EDGE)
                    check_intr = false;
                if (varg->ty == TypeInfo::Type::UNPACK)
                    return error::compiler(expr.node_info, "Internal error: not implemented");
            }

            std::vector<AstExpr> default_cargs = {};
            const std::vector<AstExpr>* param_cargs = &default_cargs;

            // Only raw lookup results and carg binds are allowed here, nothing else should get called
            if (callee->ty == TypeInfo::Type::CARGBIND)
                param_cargs = &callee->type_cargbind.cargs;
            else if (callee->ty != TypeInfo::Type::LOOKUP)
                return error::compiler(expr.node_info, "Unable to call the given type");

            if (check_def)
            {
                std::vector<std::tuple<const AstBlock*, std::map<std::string, TypeRef>, std::vector<TypeRef>>> def_matches;
                for (const AstBlock* def : callee->type_lookup.lookup.defs)
                {
                    std::map<std::string, TypeRef> cargs;
                    std::vector<TypeRef> vargs;
                    Scope sig_scope{ nullptr };
                    if (!match_def_sig(scope, sig_scope, *def, *param_cargs, param_vargs, cargs, vargs))
                        def_matches.push_back({ def, std::move(cargs), std::move(vargs) });
                }
                if (def_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to def '%' is ambiguous", callee->type_lookup.name);
                if (def_matches.size() == 1)
                {
                    // Perfect match, return it
                    std::unique_ptr<DefCall> def_call = std::make_unique<DefCall>(callee->type_lookup.lookup.ns);
                    return
                        def_call->init(*std::get<0>(def_matches.front())) ||
                        def_call->apply_args(expr.node_info, std::get<1>(def_matches.front()), std::get<2>(def_matches.front())) ||
                        def_call->codegen(scope, rets);
                }
            }
            if (check_intr)
            {
                std::vector<std::tuple<const AstBlock*, std::map<std::string, TypeRef>, std::vector<TypeRef>>> intr_matches;
                for (const AstBlock* intr : callee->type_lookup.lookup.intrs)
                {
                    std::map<std::string, TypeRef> cargs;
                    std::vector<TypeRef> vargs;
                    Scope sig_scope{ nullptr };
                    if (!match_intr_sig(scope, sig_scope, *intr, *param_cargs, param_vargs, cargs, vargs))
                        intr_matches.push_back({ intr, std::move(cargs), std::move(vargs) });
                }
                if (intr_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to intr '%' is ambiguous", callee->type_lookup.name);
                if (intr_matches.size() == 1)
                {
                    // Perfect match, return it
                    std::unique_ptr<IntrCall> intr_call = std::make_unique<IntrCall>(callee->type_lookup.lookup.ns);
                    return
                        intr_call->init(*std::get<0>(intr_matches.front())) ||
                        intr_call->apply_args(expr.node_info, std::get<1>(intr_matches.front()), std::get<2>(intr_matches.front())) ||
                        intr_call->codegen(scope, rets);
                }
            }
            // It doesn't match a def nor an intr, try for a function
            std::vector<std::tuple<const AstFn*, std::map<std::string, TypeRef>, std::vector<TypeRef>>> fn_matches;
            for (const AstFn* fn : callee->type_lookup.lookup.fns)
            {
                std::map<std::string, TypeRef> cargs;
                std::vector<TypeRef> vargs;
                Scope sig_scope{ nullptr };
                if (!match_fn_sig(scope, sig_scope, *fn, *param_cargs, param_vargs, cargs, vargs))
                    fn_matches.push_back({ fn, std::move(cargs), std::move(vargs) });
            }
            if (fn_matches.size() > 1)
                return error::compiler(expr.node_info, "Reference to fn '%' is ambiguous", callee->type_lookup.name);
            if (fn_matches.size() == 1)
            {
                // Perfect match, return it
                std::unique_ptr<FnCall> fn_call = std::make_unique<FnCall>(callee->type_lookup.lookup.ns);
                return
                    fn_call->init(*std::get<0>(fn_matches.front())) ||
                    fn_call->apply_args(expr.node_info, std::get<1>(fn_matches.front()), std::get<2>(fn_matches.front())) ||
                    fn_call->codegen(scope, rets);
            }
            return error::compiler(expr.node_info, "Unable to find a suitable overload for the varg call");
        }

        bool codegen_expr_fndecl(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_defdecl(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_kw(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            TypeRef type;
            TypeRef tmp;
            switch (expr.expr_kw)
            {
            case ExprKW::TYPE:
                tmp = type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, TypeRef());
                if (!tmp) return true;
                type = type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, tmp);
            case ExprKW::INIT:
                // the default behaviour of an init declaration is to create an init with an empty name and no configs
                tmp = type_manager->create_init(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_str(addr, "") ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            body->add_instruction(instruction::Ini(expr.node_info)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, tmp);
                break;
            case ExprKW::FTY:
                tmp = type_manager->create_fty(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_fty(addr, core::EdgeFty::F32) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_type_fty(addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    },
                    tmp);
                break;
            case ExprKW::BOOL:
                tmp = type_manager->create_bool(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_bool(addr, false) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_type_bool(addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    },
                    tmp);
                break;
            case ExprKW::INT:
                tmp = type_manager->create_int(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_int(addr, 0) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_type_int(addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    },
                    tmp);
                break;
            case ExprKW::FLOAT:
                tmp = type_manager->create_float(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_float(addr, 0) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_type_float(addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    },
                    tmp);
                break;
            case ExprKW::STR:
                tmp = type_manager->create_string(
                    TypeInfo::Category::DEFAULT,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_str(addr, "") ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_type_str(addr) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    },
                    tmp);
                break;
            case ExprKW::ARRAY:
                // Handled in codegen_expr_cargs
                return error::compiler(expr.node_info, "Invalid use of keyword 'array'");
            case ExprKW::TUPLE:
                // Handled in codegen_expr_cargs
                return error::compiler(expr.node_info, "Invalid use of keyword 'tuple'");
            case ExprKW::F16:
                type = type_manager->create_fty(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_fty(addr, core::EdgeFty::F16) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                break;
            case ExprKW::F32:
                type = type_manager->create_fty(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_fty(addr, core::EdgeFty::F32) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                break;
            case ExprKW::F64:
                type = type_manager->create_fty(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_fty(addr, core::EdgeFty::F64) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            scope.push();
                    });
                break;
            default:
                return error::compiler(expr.node_info, "Internal error: invalid keyword enum");
            }
            if (!type)
                return true;
            rets.push_back(type);
            return false;
        }

        bool codegen_expr_var(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            Scope::StackVar stack_var;
            if (!scope.at(expr.expr_string, stack_var))
            {
                // Found the variable in the scope, return the variable
                rets.push_back(stack_var.type);
                return false;
            }

            // Couldn't find it in the scope, look for it in the code module
            CodeModule::LookupResult lookup;
            if (mod->lookup({ mod->root, cg_ns.begin(), cg_ns.end() }, expr.expr_string, lookup))
            {
                // the identifier didn't match anything in the scope nor the code module.
                // This makes it an unresolved identifier.
                // Note: implicit identifiers would cause this to execute, but that case is handled in the
                // codegen_expr_assign function.  So virtual values are not needed here, its just an error
                return error::compiler(expr.node_info, "Unresolved identifier: %", expr.expr_string);
            }

            // It isn't a callee, otherwise codegen_expr_callee_var would've been called instead
            // so the var needs to match either a cargless struct or a cargless init.
            // inits shadow structs, so check the inits first for match.
            std::vector<const AstInit*> init_matches;
            for (const AstInit* init_elem : lookup.inits)
                if (init_elem->signature.cargs.size() == 0)
                    init_matches.push_back(init_elem);
            if (init_matches.size() > 1)
                return error::compiler(expr.node_info, "Reference to init '%' is ambiguous", expr.expr_string);
            if (init_matches.size() == 1)
            {
                // Perfect match, return it
                rets.push_back(type_manager->create_init(
                    TypeInfo::Category::CONST,
                    [&expr](Scope& scope) {
                        size_t addr;
                        return
                            bc->add_obj_str(addr, expr.expr_string) ||
                            body->add_instruction(instruction::New(expr.node_info, addr)) ||
                            body->add_instruction(instruction::Ini(expr.node_info)) ||
                            scope.push();
                    }));
                if (!rets.back())
                    return true;
            }

            // No init matches were found, look for struct matches
            std::vector<const AstStruct*> struct_matches;
            for (const AstStruct* struct_elem : lookup.structs)
                if (struct_elem->signature.cargs.size() == 0)
                    struct_matches.push_back(struct_elem);
            if (struct_matches.size() > 1)
                return error::compiler(expr.node_info, "Reference to struct '%' is ambiguous", expr.expr_string);
            if (struct_matches.size() == 1)
            {
                // Perfect match, return it
                return error::compiler(expr.node_info, "Internal error: structs have not been implemented yet");
            }
            
            // No struct matches were found.  It could still be a def or fn reference type of thing
            // but that hasn't been implemented yet.  More than likely its just a user error
            return error::compiler(expr.node_info, "Unable to find a cargless init or struct which matches identifier '%'", expr.expr_string);
        }

        bool codegen_expr(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            switch (expr.ty)
            {
            case ExprType::INVALID:
                return error::compiler(expr.node_info, "Internal error: invalid AstExpr type");
            case ExprType::LIT_BOOL:
                return codegen_expr_bool(scope, expr, rets);
            case ExprType::LIT_INT:
                return codegen_expr_int(scope, expr, rets);
            case ExprType::LIT_FLOAT:
                return codegen_expr_float(scope, expr, rets);
            case ExprType::LIT_STRING:
                return codegen_expr_string(scope, expr, rets);
            case ExprType::LIT_ARRAY:
                return codegen_expr_array(scope, expr, rets);
            case ExprType::LIT_TUPLE:
                return codegen_expr_tuple(scope, expr, rets);
            case ExprType::UNARY_POS:
                return codegen_expr_pos(scope, expr, rets);
            case ExprType::UNARY_NEG:
                return codegen_expr_neg(scope, expr, rets);
            case ExprType::UNARY_NOT:
                return codegen_expr_not(scope, expr, rets);
            case ExprType::UNARY_UNPACK:
                return codegen_expr_unpack(scope, expr, rets);
            case ExprType::UNARY_REF:
                return error::compiler(expr.node_info, "Invalid usage of keyword 'ref'");
            case ExprType::UNARY_CONST:
                return error::compiler(expr.node_info, "Invalid usage of keyword 'const'");
            case ExprType::UNARY_FORWARD:
                return codegen_expr_fwd(scope, expr, rets);
            case ExprType::UNARY_BACKWARD:
                return codegen_expr_bwd(scope, expr, rets);
            case ExprType::BINARY_ADD:
                return codegen_expr_add(scope, expr, rets);
            case ExprType::BINARY_SUB:
                return codegen_expr_sub(scope, expr, rets);
            case ExprType::BINARY_MUL:
                return codegen_expr_mul(scope, expr, rets);
            case ExprType::BINARY_DIV:
                return codegen_expr_div(scope, expr, rets);
            case ExprType::BINARY_MOD:
                return codegen_expr_mod(scope, expr, rets);
            case ExprType::BINARY_POW:
                return codegen_expr_pow(scope, expr, rets);
            case ExprType::BINARY_IADD:
                return codegen_expr_iadd(scope, expr, rets);
            case ExprType::BINARY_ISUB:
                return codegen_expr_isub(scope, expr, rets);
            case ExprType::BINARY_IMUL:
                return codegen_expr_imul(scope, expr, rets);
            case ExprType::BINARY_IDIV:
                return codegen_expr_idiv(scope, expr, rets);
            case ExprType::BINARY_IMOD:
                return codegen_expr_imod(scope, expr, rets);
            case ExprType::BINARY_IPOW:
                return codegen_expr_ipow(scope, expr, rets);
            case ExprType::BINARY_ASSIGN:
                return codegen_expr_assign(scope, expr, rets);
            case ExprType::BINARY_AND:
                return codegen_expr_and(scope, expr, rets);
            case ExprType::BINARY_OR:
                return codegen_expr_or(scope, expr, rets);
            case ExprType::BINARY_CMP_EQ:
                return codegen_expr_eq(scope, expr, rets);
            case ExprType::BINARY_CMP_NE:
                return codegen_expr_ne(scope, expr, rets);
            case ExprType::BINARY_CMP_GT:
                return codegen_expr_gt(scope, expr, rets);
            case ExprType::BINARY_CMP_LT:
                return codegen_expr_lt(scope, expr, rets);
            case ExprType::BINARY_CMP_GE:
                return codegen_expr_ge(scope, expr, rets);
            case ExprType::BINARY_CMP_LE:
                return codegen_expr_le(scope, expr, rets);
            case ExprType::BINARY_CAST:
                return codegen_expr_cast(scope, expr, rets);
            case ExprType::INDEX:
                return codegen_expr_idx(scope, expr, rets);
            case ExprType::DOT:
                return codegen_expr_dot(scope, expr, rets);
            case ExprType::VAR_DECL:
                return codegen_expr_decl(scope, expr, rets);
            case ExprType::CARGS_CALL:
                return codegen_expr_cargs(scope, expr, rets);
            case ExprType::VARGS_CALL:
                return codegen_expr_vargs(scope, expr, rets);
            case ExprType::FN_DECL:
                return codegen_expr_fndecl(scope, expr, rets);
            case ExprType::DEF_DECL:
                return codegen_expr_defdecl(scope, expr, rets);
            case ExprType::KW:
                return codegen_expr_kw(scope, expr, rets);
            case ExprType::VAR:
                return codegen_expr_var(scope, expr, rets);
            }
            return error::compiler(expr.node_info, "Internal error: unknown AstExpr type");
        }

        bool codegen_exit(Scope& scope, const AstNodeInfo& type)
        {
            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, scope.get_parent()))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body->add_instruction(instruction::Pop(type, var.ptr)))
                    return true;
            }
            scope.pop(vars.size());
            return false;
        }

        bool codegen_line_break(Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "break statements are not allowed outside a looping structure");

            size_t local_sz = 0;
            if (scope.local_size(local_sz, loop_ctx->scope))
                return true;
            for (size_t i = 0; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 0)))
                    return true;
            return body->add_instruction(instruction::Jmp(line.node_info, loop_ctx->break_label));
        }

        bool codegen_line_continue(Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "continue statements are not allowed outside a looping structure");

            size_t local_sz = 0;
            if (scope.local_size(local_sz, loop_ctx->scope))
                return true;
            for (size_t i = 0; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 0)))
                    return true;
            return body->add_instruction(instruction::Jmp(line.node_info, loop_ctx->cont_label));
        }

        bool codegen_line_export(Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_extern(Scope& scope, const AstLine& line)
        {
            Scope::StackVar var;
            size_t name_addr;
            TypeRef init;
            if (scope.at(line.line_extern.var_name, var) ||
                bc->add_obj_str(name_addr, line.line_extern.var_name) ||
                codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, line.line_extern.init_expr, init)
                ) return true;
            if (init->ty != TypeInfo::Type::INIT)
                return error::compiler(line.line_extern.init_expr.node_info, "Expected an init, recieved %", init->to_string());
            if (body_type != BodyType::DEF)
                return error::compiler(line.node_info, "The keyword 'extern' can only be used inside def blocks");
            if (scope.at("~block", var) ||
                body->add_instruction(instruction::Dup(line.node_info, var.ptr)) ||
                scope.push() ||
                scope.at(line.line_extern.var_name, var) ||
                body->add_instruction(instruction::Dup(line.node_info, var.ptr)) ||
                scope.push() ||
                init->codegen(scope) ||
                body->add_instruction(instruction::SIni(line.node_info)) ||
                body->add_instruction(instruction::New(line.node_info, name_addr)) ||
                body->add_instruction(instruction::BkExt(line.node_info)) ||
                scope.pop(3)
                ) return error::compiler(line.node_info, "Unable to compile extern");
            return false;
        }

        bool codegen_line_raise(Scope& scope, const AstLine& line)
        {
            TypeRef ret;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, line.line_func.expr, ret))
                return true;
            if (ret->ty != TypeInfo::Type::STR)
                return error::compiler(line.line_func.expr.node_info, "raise expected string, recieved %", ret->to_string());
            return
                ret->codegen(scope) ||
                body->add_instruction(instruction::Err(line.node_info)) ||
                scope.pop();
        }

        bool codegen_line_print(Scope& scope, const AstLine& line)
        {
            TypeRef ret;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, line.line_func.expr, ret))
                return true;
            if (ret->ty != TypeInfo::Type::STR)
                return error::compiler(line.line_func.expr.node_info, "print expected string, recieved %", ret->to_string());
            return
                ret->codegen(scope) ||
                body->add_instruction(instruction::Dsp(line.node_info)) ||
                scope.pop();
        }

        bool codegen_line_retdef(Scope& scope, const AstLine& line)
        {
            if (!ret_label)
                return error::compiler(line.node_info, "Internal error: ret_label was nullptr");
            // Getting the rets
            std::vector<TypeRef> rets;
            size_t local_sz = 0;
            if (codegen_expr(scope, line.line_func.expr, rets) ||
                scope.local_size(local_sz, nullptr)
                ) return true;
            if (rets.size() == 1 && rets.back()->ty == TypeInfo::Type::TUPLE)
                // Expanding the tuple
                rets = rets.back()->type_tuple.elems;
            // Checking the number of provided rets against the signature
            if (rets.size() != body_sig.block_sig->rets.size())
                return error::compiler(line.node_info, "Expected % return value(s), recieved %", body_sig.block_sig->rets.size(), rets.size());
            // Codegening all the rets onto the stack
            for (TypeRef ret : rets)
            {
                if (ret->ty != TypeInfo::Type::TENSOR)
                    return error::compiler(line.node_info, "The return values from a def must be all tensors");
                if (ret->cat == TypeInfo::Category::VIRTUAL)
                    return error::compiler(line.node_info, "The return values from a def must be non-virtual");

                if (ret->codegen(scope))
                    return true;
            }
            // Popping everything in the scope except the rets and args
            for (size_t i = body_sig.block_sig->cargs.size() + body_sig.block_sig->vargs.size() + 1; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, rets.size())))
                    return true;
            return body->add_instruction(instruction::Jmp(line.node_info, *ret_label));
        }

        bool codegen_line_retfn(Scope& scope, const AstLine& line)
        {
            // Codegening each of the return types from the signature to compare against
            std::vector<TypeRef> sig_rets;
            for (const AstExpr& expr : body_sig.fn_sig->rets)
            {
                TypeRef ret;
                if (codegen_expr_single_ret<TypeInfo::AllowAll>(scope, expr, ret))
                    return true;
                if (ret->ty != TypeInfo::Type::TYPE && ret->ty != TypeInfo::Type::DLTYPE)
                    return error::compiler(expr.node_info, "The return expression did not resolve to a type");
                sig_rets.push_back(ret);
            }
            // Getting the rets
            std::vector<TypeRef> rets;
            size_t local_sz = 0;
            if (codegen_expr(scope, line.line_func.expr, rets) ||
                scope.local_size(local_sz, nullptr)  // None of the remaining code should add stuff to the scope
                ) return true;
            if (rets.size() == 1 && rets.back()->ty == TypeInfo::Type::TUPLE)
                // Expanding the tuple
                rets = rets.back()->type_tuple.elems;
            // Type checking each of the rets
            if (rets.size() != sig_rets.size())
                return error::compiler(line.node_info, "Expected % return value(s), recieved %", sig_rets.size(), rets.size());
            for (size_t i = 0; i < rets.size(); i++)
            {
                if (sig_rets[i]->ty == TypeInfo::Type::TYPE)
                {
                    // I can do compile time check for the return value
                    if (sig_rets[i]->type_type.base != rets[i])
                        return error::compiler(line.node_info, error::format("Expected type % for return value %, but recieved type %",
                            sig_rets[i]->type_type.base->to_string(), i + 1, rets[i]->to_string()));
                }
                else
                {
                    assert(sig_rets[i]->ty == TypeInfo::Type::DLTYPE);
                    // I have to embed a check into the runtime
                    std::string fty_end_label = label_prefix(line.node_info) + "_ret" + std::to_string(i) + "_fty_end";
                    std::string shp_end_label = label_prefix(line.node_info) + "_ret" + std::to_string(i) + "_shp_end";
                    if (rets[i]->ty == TypeInfo::Type::TENSOR)
                    {
                        size_t fty_errmsg, shp_errmsg;
                        if (bc->add_obj_str(fty_errmsg, error::format("Tensor fty mismatch found in return value %", i + 1)) ||
                            bc->add_obj_str(shp_errmsg, error::format("Tensor shape mismatch found in return value %", i + 1))
                            ) return true;

                        // Checking the fty
                        size_t fty_addr;
                        if (bc->add_type_int(fty_addr) ||
                            rets[i]->codegen(scope) ||  // This should only be a dup, so its not worth dupping the codegen result
                            body->add_instruction(instruction::Tfty(line.node_info)) ||
                            sig_rets[i]->type_dltype.fp->codegen(scope) ||
                            body->add_instruction(instruction::New(line.node_info, fty_addr)) ||
                            body->add_instruction(instruction::Eq(line.node_info)) ||
                            body->add_instruction(instruction::Brt(line.node_info, fty_end_label)) ||
                            scope.pop(2) ||
                            // Error condition instructions
                            body->add_instruction(instruction::New(line.node_info, fty_errmsg)) ||
                            body->add_instruction(instruction::Err(line.node_info)) ||
                            body->add_label(line.node_info, fty_end_label)
                            ) return true;

                        // Checking the shape
                        size_t int_addr;
                        if (bc->add_type_int(int_addr) ||
                            rets[i]->codegen(scope) ||  // This should only be a dup, so its not worth dupping the codegen result
                            body->add_instruction(instruction::Tshp(line.node_info)) ||
                            sig_rets[i]->type_dltype.shape->codegen(scope) ||
                            body->add_instruction(instruction::New(line.node_info, int_addr)) ||
                            body->add_instruction(instruction::Arr(line.node_info)) ||
                            body->add_instruction(instruction::Eq(line.node_info)) ||
                            body->add_instruction(instruction::Brt(line.node_info, shp_end_label)) ||
                            scope.pop(2) ||
                            // Error condition instructions
                            body->add_instruction(instruction::New(line.node_info, shp_errmsg)) ||
                            body->add_instruction(instruction::Err(line.node_info)) ||
                            body->add_label(line.node_info, shp_end_label)
                            ) return true;
                        continue;
                    }
                    return error::compiler(line.node_info, "Expected a tensor for return value %, recieved %", i + 1, rets[i]->to_string());
                }
            }
#ifdef _DEBUG
            // Making sure the above code didn't do anything to the scope
            size_t debug_sz = 0;
            if (scope.local_size(debug_sz, nullptr))
                return true;
            assert(debug_sz == local_sz);
#endif
            // Codegening all the rets onto the stack
            for (TypeRef ret : rets)
                if (ret->codegen(scope))
                    return true;
            // Popping everything in the scope except the rets and args
            for (size_t i = body_sig.fn_sig->cargs.size() + body_sig.fn_sig->vargs.size() + 1; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, rets.size())))
                    return true;
            return body->add_instruction(instruction::Ret(line.node_info));
        }

        bool codegen_line_retintr(Scope& scope, const AstLine& line)
        {
            if (!ret_label)
                return error::compiler(line.node_info, "Internal error: ret_label was nullptr");
            // Getting the rets
            std::vector<TypeRef> rets;
            size_t local_sz = 0;
            if (codegen_expr(scope, line.line_func.expr, rets) ||
                scope.local_size(local_sz, nullptr)
                ) return true;
            if (rets.size() == 1 && rets.back()->ty == TypeInfo::Type::TUPLE)
                // Expanding the tuple
                rets = rets.back()->type_tuple.elems;
            // Checking the number of provided rets against the signature
            if (rets.size() != body_sig.block_sig->rets.size())
                return error::compiler(line.node_info, "Expected % return value(s), recieved %", body_sig.block_sig->rets.size(), rets.size());
            // Codegening all the rets onto the stack and checking the individual types as I go
            for (TypeRef ret : rets)
            {
                if (ret->ty != TypeInfo::Type::EDGE)
                    return error::compiler(line.node_info, "The return values from an intr must be all edges");
                if (ret->cat == TypeInfo::Category::VIRTUAL)
                    return error::compiler(line.node_info, "The return values from an intr must be non-virtual");

                if (ret->codegen(scope))
                    return true;
            }
            // Popping everything in the scope except the rets, args, and the intr node
            for (size_t i = body_sig.block_sig->cargs.size() + body_sig.block_sig->vargs.size() + 2; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, rets.size())))
                    return true;
            return body->add_instruction(instruction::Jmp(line.node_info, *ret_label));
        }

        bool codegen_line_return(Scope& scope, const AstLine& line)
        {
            switch (body_type)
            {
            case BodyType::INVALID:
                return error::compiler(line.node_info, "Internal error: body_type enum was INVALID");
            case BodyType::STRUCT:
                return error::compiler(line.node_info, "Keyword 'return' is not allowed in a struct definition");
            case BodyType::DEF:
                return codegen_line_retdef(scope, line);
            case BodyType::FN:
                return codegen_line_retfn(scope, line);
            case BodyType::INTR:
                return codegen_line_retintr(scope, line);
            }
            return error::compiler(line.node_info, "Internal error: body_type enum was out of range");
        }

        bool codegen_line_branch(Scope& scope, const AstLine& line, const std::string& end_label)
        {
            Scope block_scope{ &scope };
            TypeRef ret;
            size_t local_sz = 0;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(block_scope, line.line_branch.cond, ret) ||
                block_scope.local_size(local_sz, block_scope.get_parent())
                ) return true;
            if (ret->ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a boolean value");
            if (ret->codegen(block_scope))
                return true;
            // Clearing out any stack variables created during the condition evaluation (except for the condition itself)
            for (size_t i = 1; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 1)) ||
                    block_scope.pop()
                    ) return true;
            std::string false_branch = label_prefix(line.node_info) + "false_branch";
            if (body->add_instruction(instruction::Brf(line.node_info, false_branch)) ||
                block_scope.pop() ||
                codegen_lines(block_scope, line.line_branch.body) ||
                block_scope.local_size(local_sz, block_scope.get_parent())
                ) return true;
            // Clearing out any stack variables created in the body of the branch
            for (size_t i = 1; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                    scope.pop()
                    ) return true;
            // Ending off the branch block
            return
                body->add_instruction(instruction::Jmp(line.node_info, end_label)) ||
                body->add_label(line.node_info, false_branch);
        }

        bool codegen_line_while(Scope& scope, const AstLine& line)
        {
            LoopContext* old_ctx = loop_ctx;
            Scope block_scope{ &scope };
            std::string loop_start = label_prefix(line.node_info) + "loop_start";
            std::string loop_end = label_prefix(line.node_info) + "loop_end";
            TypeRef ret;
            if (body->add_label(line.node_info, loop_start) ||
                codegen_expr_single_ret<TypeInfo::NonVirtual>(block_scope, line.line_branch.cond, ret)
                ) return true;
            LoopContext new_ctx = { &scope, loop_start, loop_end };
            loop_ctx = &new_ctx;
            if (ret->ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a boolean value");
            size_t local_sz = 0;
            if (block_scope.local_size(local_sz, block_scope.get_parent()) ||
                ret->codegen(block_scope)
                ) return error::compiler(line.node_info, "Unable to compile while loop condition");
            // Clearing the scope of any stack variables creating in the while loop condition (besides the condition itself)
            for (size_t i = 1; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 1)) ||
                    block_scope.pop()
                    ) return true;
            // While loop condition and body
            if (body->add_instruction(instruction::Brf(line.node_info, loop_end)) ||
                block_scope.pop() ||
                codegen_lines(block_scope, line.line_branch.body) ||
                block_scope.local_size(local_sz, block_scope.get_parent())
                ) return error::compiler(line.node_info, "Unable to compile while loop body");
            // Clearing the scope of any declarations present in the body
            for (size_t i = 0; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                    block_scope.pop()
                    ) return true;
            if (body->add_instruction(instruction::Jmp(line.node_info, loop_start)) ||
                body->add_label(line.node_info, loop_end)
                ) return true;
            loop_ctx = old_ctx;
            return false;
        }

        bool codegen_line_for(Scope& scope, const AstLine& line)
        {
            LoopContext* old_ctx = loop_ctx;
            TypeRef ret;
            if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, line.line_for.iter, ret))
                return true;
            if (ret->ty != TypeInfo::Type::ARRAY)
                return error::compiler(line.line_for.iter.node_info, "For loops are only able to iterate through array types");
            Scope block_scope{ &scope };
            std::string loop_start = label_prefix(line.node_info) + "_loop_start";
            std::string loop_cont = label_prefix(line.node_info) + "_loop_continue";
            std::string loop_end = label_prefix(line.node_info) + "_loop_end";
            size_t zero_addr, one_addr, int_addr;
            size_t prev_sz = 0;
            if (bc->add_obj_int(zero_addr, 0) ||
                bc->add_obj_int(one_addr, 1) ||
                bc->add_type_int(int_addr) ||
                // Setup the for loop (stack will be [iter, len(iter), i])
                // None of this setup is in the block scope, its in the parent scope and is cleaned seperately
                scope.local_size(prev_sz, scope.get_parent()) ||
                ret->codegen(scope) ||
                body->add_instruction(instruction::Dup(line.node_info, 0)) ||
                body->add_instruction(instruction::Nul(line.node_info)) ||
                body->add_instruction(instruction::Arr(line.node_info)) ||
                body->add_instruction(instruction::Len(line.node_info)) ||
                body->add_instruction(instruction::New(line.node_info, zero_addr)) ||
                body->add_instruction(instruction::New(line.node_info, int_addr)) ||
                body->add_instruction(instruction::Cpy(line.node_info)) ||
                scope.push(2) ||
                // for loop condition
                body->add_label(line.node_info, loop_start) ||
                body->add_instruction(instruction::Dup(line.node_info, 1)) ||
                body->add_instruction(instruction::Dup(line.node_info, 1)) ||
                body->add_instruction(instruction::New(line.node_info, int_addr)) ||
                body->add_instruction(instruction::Eq(line.node_info)) ||
                body->add_instruction(instruction::Brt(line.node_info, loop_end))
                ) return error::compiler(line.node_info, "Unable to compile for loop collection");
            
            // assignment
            if (line.line_for.decl.ty == ExprType::VAR)
            {
                // implicit declaration
                if (block_scope.contains(line.line_for.decl.expr_string))
                    return error::compiler(line.line_for.decl.node_info, "Unable to use an already defined variable as in for loop");
                // Creating a type for the iterator
                TypeRef elem_type = type_manager->duplicate(
                    TypeInfo::Category::DEFAULT,
                    [&line](Scope& scope) -> bool {
                        Scope::StackVar var;
                        return
                            scope.at(line.line_for.decl.expr_string, var) ||
                            body->add_instruction(instruction::Dup(line.node_info, var.ptr)) ||
                            scope.push();
                    }, ret->type_array.elem);
                if (!elem_type)
                    return true;
                // Indexing into iter and putting the variable onto the stack
                if (body->add_instruction(instruction::Dup(line.node_info, 2)) ||
                    body->add_instruction(instruction::Dup(line.node_info, 1)) ||
                    body->add_instruction(instruction::Nul(line.node_info)) ||
                    body->add_instruction(instruction::Arr(line.node_info)) ||
                    body->add_instruction(instruction::Idx(line.node_info)) ||
                    block_scope.push() ||
                    block_scope.add(line.line_for.decl.expr_string, elem_type, line.node_info)
                    ) return true;
            }
            else
                return error::compiler(line.line_for.decl.node_info, "Internal error: not implemented");
            
            // for loop body
            LoopContext new_ctx = { &scope, loop_cont, loop_end };
            loop_ctx = &new_ctx;
            size_t local_sz = 0;
            if (codegen_lines(block_scope, line.line_for.body) ||
                block_scope.local_size(local_sz, block_scope.get_parent())
                ) return error::compiler(line.node_info, "Unable to compile for loop body");
            loop_ctx = old_ctx;
            
            // Clearing the stack of all variables declared in the body of the loop (this will include the assignment)
            for (size_t i = 0; i < local_sz; i++)
                if (body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                    scope.pop()
                    ) return true;
            
            if (body->add_label(line.node_info, loop_cont) ||
                // increment the array index
                body->add_instruction(instruction::Dup(line.node_info, 0)) ||
                body->add_instruction(instruction::New(line.node_info, one_addr)) ||
                body->add_instruction(instruction::New(line.node_info, int_addr)) ||
                body->add_instruction(instruction::IAdd(line.node_info)) ||
                // loop back
                body->add_instruction(instruction::Jmp(line.node_info, loop_start)) ||
                body->add_label(line.node_info, loop_end) ||
                // getting the current size of the current scope
                scope.local_size(local_sz, scope.get_parent())
                ) return true;

            // unwind the stack setup
            if (body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                scope.pop(3)
                ) return true;
            return false;
        }

        bool codegen_line_expr(Scope& scope, const AstLine& line)
        {
            // building out the rets for the expression
            std::vector<TypeRef> rets;
            if (codegen_expr(scope, line.line_expr.line, rets))
                return true;

            // actually codegening and getting rid of the immediate rets, not local variables
            for (TypeRef type : rets)
            {
                if (type->codegen(scope) ||
                    body->add_instruction(instruction::Pop(line.node_info, 0)) ||
                    scope.pop()
                    ) return true;
            }
            return false;
        }

        bool codegen_line(Scope& scope, const AstLine& line)
        {
            switch (line.ty)
            {
            case LineType::INVALID:
                return error::compiler(line.node_info, "Internal error: invalid AstLine type");
            case LineType::BREAK:
                return codegen_line_break(scope, line);
            case LineType::CONTINUE:
                return codegen_line_continue(scope, line);
            case LineType::EXPORT:
                return codegen_line_export(scope, line);
            case LineType::EXTERN:
                return codegen_line_extern(scope, line);
            case LineType::RAISE:
                return codegen_line_raise(scope, line);
            case LineType::PRINT:
                return codegen_line_print(scope, line);
            case LineType::RETURN:
                return codegen_line_return(scope, line);
            case LineType::IF:
                return error::compiler(line.node_info, "Internal error: recieved dependent line type 'if' in codegen_line");
            case LineType::ELIF:
                return error::compiler(line.node_info, "Found elif statement without a matching if statement");
            case LineType::ELSE:
                return error::compiler(line.node_info, "Found else statement without a matching if statement");
            case LineType::WHILE:
                return codegen_line_while(scope, line);
            case LineType::FOR:
                return codegen_line_for(scope, line);
            case LineType::EXPR:
                return codegen_line_expr(scope, line);
            case LineType::EVALMODE:
                return error::compiler(line.node_info, "Internal error: not implemented");
            default:
                return error::compiler(line.node_info, "Internal error: unknown AstLine type");
            }
        }

        bool codegen_lines(Scope& scope, const std::vector<AstLine>& lines)
        {
            if (lines.size() == 0)
                return false;  // If no lines are there, no need to codegen anything

            bool ret = false;
            for (size_t i = 0; i < lines.size(); i++)
            {
                if (lines[i].ty == LineType::IF)
                {
                    std::string end_label = label_prefix(lines[i].node_info) + "branch_end";
                    if (codegen_line_branch(scope, lines[i], end_label))
                        return true;
                    i++;
                    while (i < lines.size() && lines[i].ty == LineType::ELIF)
                    {
                        if (codegen_line_branch(scope, lines[i], end_label))
                            return true;
                        i++;
                    }
                    if (i < lines.size() && lines[i].ty == LineType::ELSE)
                    {
                        if (codegen_lines(scope, lines[i].line_block.body))
                            return true;
                        i++;
                    }
                    i--;
                    if (body->add_label(lines[i].node_info, end_label))
                        return true;
                }
                else
                    ret = codegen_line(scope, lines[i]) || ret;
            }
            return ret;
        }

        bool proc_name_struct(std::string& name, const std::vector<std::string>& ns, const AstStruct& ast_struct)
        {
            return error::compiler(ast_struct.node_info, "Internal error: not implemented");
        }

        bool proc_name_fn(std::string& name, const std::vector<std::string>& ns, const AstFn& ast_fn)
        {
            std::stringstream fn_name;
            fn_name << "fn_" << ns.size() << "_";
            for (const auto& ns_name : ns)
                fn_name << ns_name << "_";
            Scope scope{ nullptr };
            fn_name << ast_fn.signature.cargs.size() << "_";
            for (const auto& arg : ast_fn.signature.cargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
                fn_name << type->encode() << "_";
            }
            fn_name << ast_fn.signature.vargs.size() << "_";
            for (const auto& arg : ast_fn.signature.vargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg))
                    return true;
                fn_name << type->encode() << "_";
            }
            fn_name << ast_fn.signature.name;
            name = fn_name.str();
            if (name.find('-') != std::string::npos)
                return error::compiler(ast_fn.node_info, "Found invalid argument while encoding fn: %", name);
            return false;
        }

        bool proc_name_def(std::string& name, const std::vector<std::string>& ns, const AstBlock& ast_def)
        {
            std::stringstream def_name;
            def_name << "def_" << ns.size() << "_";
            for (const auto& ns_name : ns)
                def_name << ns_name << "_";
            Scope scope{ nullptr };
            def_name << ast_def.signature.cargs.size() << "_";
            for (const AstArgDecl& arg : ast_def.signature.cargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
                def_name << type->encode() << "_";
            }
            def_name << ast_def.signature.vargs.size() << "_" << ast_def.signature.name;
            name = def_name.str();
            if (name.find('-') != std::string::npos)
                return error::compiler(ast_def.node_info, "Found invalid argument while encoding def: %", name);
            return false;
        }

        bool proc_name_intr(std::string& name, const std::vector<std::string>& ns, const AstBlock& ast_intr)
        {
            std::stringstream intr_name;
            intr_name << "intr_" << ns.size() << "_";
            for (const auto& ns_name : ns)
                intr_name << ns_name << "_";
            Scope scope{ nullptr };
            intr_name << ast_intr.signature.cargs.size() << "_";
            for (const auto& arg : ast_intr.signature.cargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
                intr_name << type->encode() << "_";
            }
            intr_name << ast_intr.signature.vargs.size() << "_" << ast_intr.signature.name;
            name = intr_name.str();
            if (name.find('-') != std::string::npos)
                return error::compiler(ast_intr.node_info, "Found invalid argument while encoding intr: %", name);
            return false;
        }

        bool codegen_struct(const std::string& name, const AstStruct& ast_struct, const std::vector<std::string>& ns)
        {
            body_type = BodyType::STRUCT;
            ByteCodeBody body{ ast_struct.node_info };

            return error::compiler(ast_struct.node_info, "Not implemented");
        }

        bool codegen_proc_safety(const AstNodeInfo& node_info)
        {
            // During proper code execution, this point should be unreachable.
            // If this point is reached during execution, it will fall through to the next function which is undefined
            // and will cause the interpreter to start bahaving unpredictably which would be very difficult to debug.
            // For safety, a raise statement is added manually by the compiler to generate a runtime error
            // if the programmer made a mistake.  If this point was in fact unreachable,
            // the raise will never get run and will not have any consequences (besides moving labels around in the bytecode)
            size_t addr;
            return
                bc->add_obj_str(addr, "Reached the end of the procedure without returning") ||
                body->add_instruction(instruction::New(node_info, addr)) ||
                body->add_instruction(instruction::Err(node_info));
        }

        bool codegen_func(const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns)
        {
            body_type = BodyType::FN;
            body_sig.fn_sig = &ast_fn.signature;
            ByteCodeBody fn_body{ ast_fn.node_info };
            TypeManager fn_type_manager{};
            body = &fn_body;
            type_manager = &fn_type_manager;
            cg_ns = ns;

            Scope scope{ nullptr };
            std::string fn_name;
            if (proc_name_fn(fn_name, ns, ast_fn))
                return true;
            
            for (const auto& arg : ast_fn.signature.cargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            for (const auto& arg : ast_fn.signature.vargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            TypeRef blk_ty = type_manager->create_block(
                TypeInfo::Category::CONST,
                [&ast_fn](Scope& scope) -> bool {
                    Scope::StackVar var;
                    return
                        scope.at("~block", var) ||
                        body->add_instruction(instruction::Dup(ast_fn.node_info, var.ptr)) ||
                        scope.push();
                });
            if (!blk_ty)
                return true;
            if (scope.add("~block", blk_ty, ast_fn.node_info))
                return true;

            if (bc->has_proc(fn_name))
                // TODO: provide a better error message
                return error::compiler(ast_fn.node_info, "fn block conflict found");
            if (codegen_lines(scope, ast_fn.body) ||
                codegen_proc_safety(ast_fn.node_info) ||
                bc->add_block(fn_name, *body)
                ) return error::compiler(ast_fn.node_info, "Unable to compile body of function");
            return false;
        }

        bool codegen_def(const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns)
        {
            body_type = BodyType::DEF;
            body_sig.block_sig = &ast_def.signature;
            ByteCodeBody def_body{ ast_def.node_info };
            TypeManager def_type_manager{};
            body = &def_body;
            type_manager = &def_type_manager;
            cg_ns = ns;

            Scope scope{ nullptr };
            std::string def_name;
            if (proc_name_def(def_name, ns, ast_def))
                return true;

            // Constructing the scope
            for (const auto& arg : ast_def.signature.cargs)
            {
                // Adding each of the cargs onto the stack
                // At the bytecode level, packed arguments are a single value
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            for (const auto& arg : ast_def.signature.vargs)
            {
                // Adding each of the vargs onto the stack
                TypeRef type;
                if (arg_type(type, scope, arg))
                    return true;
                if (type->ty != TypeInfo::Type::TENSOR)
                    return error::compiler(arg.node_info, "def vargs must be tensors");
                if (scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            // Adding the block reference onto the stack
            TypeRef blk_ty = type_manager->create_block(
                TypeInfo::Category::CONST,
                [&ast_def](Scope& scope) -> bool {
                    Scope::StackVar var;
                    return
                        scope.at("~block", var) ||
                        body->add_instruction(instruction::Dup(ast_def.node_info, var.ptr)) ||
                        scope.push();
                });
            if (!blk_ty)
                return true;
            if (scope.add("~block", blk_ty, ast_def.node_info))
                return true;

            // Bytecode for creating the new block
            size_t def_name_addr;
            if (bc->add_obj_str(def_name_addr, name) ||
                body->add_instruction(instruction::New(ast_def.node_info, def_name_addr)) ||
                body->add_instruction(instruction::Blk(ast_def.node_info)) ||
                body->add_instruction(instruction::BkPrt(ast_def.node_info))
                ) return true;

            // Bytecode for the block configurations
            for (const auto& arg : ast_def.signature.cargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve def carg '%'", arg.var_name);
                // Assume it's type can be a runtime object.
                // If it can't, it shouldn't have been in the cargs, so to_obj will generate an error
                size_t addr;
                TypeRef ty;
                if (body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                    var.type->to_obj(arg.node_info, ty) ||
                    ty->codegen(scope) ||
                    bc->add_obj_str(addr, arg.var_name) ||
                    body->add_instruction(instruction::New(arg.node_info, addr)) ||
                    body->add_instruction(instruction::BkCfg(arg.node_info)) ||
                    scope.pop()
                    ) return true;
            }

            // Bytecode for adding block inputs
            for (const auto& arg : ast_def.signature.vargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve def varg '%'", arg.var_name);
                assert(var.type->ty == TypeInfo::Type::TENSOR);
                size_t addr;
                if (body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                    bc->add_obj_str(addr, arg.var_name) ||
                    body->add_instruction(instruction::New(arg.node_info, addr)) ||
                    body->add_instruction(instruction::BkInp(arg.node_info))
                    ) return true;
            }

            // Determining the name for the return label
            std::string def_ret_label = label_prefix(ast_def.node_info) + "_return";
            ret_label = &def_ret_label;
            if (codegen_lines(scope, ast_def.body))  // generating the code for the body
                return error::compiler(ast_def.node_info, "Unable to compile body of def block");
            ret_label = nullptr;

            // Bytecode for the return subroutine

            if (codegen_proc_safety(ast_def.node_info) ||
                body->add_label(ast_def.node_info, def_ret_label) ||
                body->add_instruction(instruction::Dup(ast_def.node_info, ast_def.signature.rets.size()))  // grabbing a reference to the block
                ) return true;

            // Adding block outputs
            for (size_t i = 0; i < ast_def.signature.rets.size(); i++)
            {
                size_t addr;
                if (body->add_instruction(instruction::Dup(ast_def.node_info, ast_def.signature.rets.size() - i)) ||  // no -1 to account for the block reference
                    bc->add_obj_str(addr, ast_def.signature.rets[i]) ||
                    body->add_instruction(instruction::New(ast_def.node_info, addr)) ||
                    body->add_instruction(instruction::BkOut(ast_def.node_info))
                    ) return true;
            }

            if (bc->has_proc(def_name))
                // TODO: provide a better error message
                return error::compiler(ast_def.node_info, "def block conflict found");
            return
                body->add_instruction(instruction::Pop(ast_def.node_info, 0)) ||  // popping the block reference off (from the output edge configuration, not the argument)
                body->add_instruction(instruction::Ret(ast_def.node_info)) ||
                bc->add_block(def_name, *body);
        }

        bool codegen_intr(const std::string& name, const AstBlock& ast_intr, const std::vector<std::string>& ns)
        {
            body_type = BodyType::INTR;
            body_sig.block_sig = &ast_intr.signature;
            ByteCodeBody intr_body{ ast_intr.node_info };
            TypeManager intr_type_manager{};
            body = &intr_body;
            type_manager = &intr_type_manager;
            cg_ns = ns;

            Scope scope{ nullptr };
            std::string intr_name;
            if (proc_name_intr(intr_name, ns, ast_intr))
                return true;

            // Constructing the scope
            for (const auto& arg : ast_intr.signature.cargs)
            {
                // Adding each of the cargs onto the stack
                // At the bytecode level, packed arguments are a single value
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            for (const auto& arg : ast_intr.signature.vargs)
            {
                // Adding each of the vargs onto the stack
                TypeRef type;
                if (arg_type(type, scope, arg))
                    return true;
                if (type->ty != TypeInfo::Type::EDGE)
                    return error::compiler(arg.node_info, "intr vargs must be edges");
                if (scope.add(arg.var_name, type, arg.node_info) ||
                    scope.push()
                    ) return true;
            }
            // Adding the block reference onto the stack
            TypeRef blk_ty = type_manager->create_block(
                TypeInfo::Category::CONST,
                [&ast_intr](Scope& scope) -> bool {
                    Scope::StackVar var;
                    return
                        scope.at("~block", var) ||
                        body->add_instruction(instruction::Dup(ast_intr.node_info, var.ptr)) ||
                        scope.push();
                });
            if (!blk_ty || scope.add("~block", blk_ty, ast_intr.node_info))
                return true;

            // Making the node type
            TypeRef nde_ty = type_manager->create_node(
                TypeInfo::Category::CONST,
                [&ast_intr](Scope& scope) -> bool {
                    Scope::StackVar var;
                    return
                        scope.at("~node", var) ||
                        body->add_instruction(instruction::Dup(ast_intr.node_info, var.ptr)) ||
                        scope.push();
                });
            if (!nde_ty)
                return true;

            // Bytecode for creating the node
            size_t intr_name_addr;
            if (bc->add_obj_str(intr_name_addr, name) ||
                body->add_instruction(instruction::Dup(ast_intr.node_info, 0)) ||
                body->add_instruction(instruction::New(ast_intr.node_info, intr_name_addr)) ||
                body->add_instruction(instruction::Nde(ast_intr.node_info)) ||
                body->add_instruction(instruction::NdPrt(ast_intr.node_info)) ||
                scope.push() ||
                scope.add("~node", nde_ty, ast_intr.node_info)
                ) return true;

            // Bytecode for the node configurations
            for (const auto& arg : ast_intr.signature.cargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve intr carg '%'", arg.var_name);
                // Assume it's type can be a runtime object.
                // If it can't, it shouldn't have been in the cargs, so to_obj will generate an error
                size_t addr;
                TypeRef ty;
                if (body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                    var.type->to_obj(arg.node_info, ty) ||
                    ty->codegen(scope) ||
                    bc->add_obj_str(addr, arg.var_name) ||
                    body->add_instruction(instruction::New(arg.node_info, addr)) ||
                    body->add_instruction(instruction::NdCfg(arg.node_info)) ||
                    scope.pop()
                    ) return true;
            }

            // Bytecode for adding node inputs
            for (const auto& arg : ast_intr.signature.vargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve intr varg '%'", arg.var_name);
                assert(var.type->ty == TypeInfo::Type::EDGE);
                size_t addr;
                if (body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                    bc->add_obj_str(addr, arg.var_name) ||
                    body->add_instruction(instruction::New(arg.node_info, addr)) ||
                    body->add_instruction(instruction::NdInp(arg.node_info))
                    ) return true;
            }

            // Determining the name for the return label
            std::string intr_ret_label = label_prefix(ast_intr.node_info) + "_return";
            ret_label = &intr_ret_label;
            if (codegen_lines(scope, ast_intr.body))  // generating the code for the body
                return error::compiler(ast_intr.node_info, "Unable to compile body of intrinsic");
            ret_label = nullptr;

            // Bytecode for the return subroutine

            if (codegen_proc_safety(ast_intr.node_info) ||
                body->add_label(ast_intr.node_info, intr_ret_label) ||
                body->add_instruction(instruction::Dup(ast_intr.node_info, ast_intr.signature.rets.size()))  // grabbing a reference to the node
                ) return true;

            // Adding node outputs
            for (size_t i = 0; i < ast_intr.signature.rets.size(); i++)
            {
                size_t addr;
                if (body->add_instruction(instruction::Dup(ast_intr.node_info, ast_intr.signature.rets.size() - i)) ||  // no -1 to account for the node reference
                    bc->add_obj_str(addr, ast_intr.signature.rets[i]) ||
                    body->add_instruction(instruction::New(ast_intr.node_info, addr)) ||
                    body->add_instruction(instruction::NdOut(ast_intr.node_info))
                    ) return true;
            }

            if (bc->has_proc(intr_name))
                // TODO: provide a better error message
                return error::compiler(ast_intr.node_info, "intr proc conflict found");
            return
                body->add_instruction(instruction::Pop(ast_intr.node_info, 0)) ||  // popping the node reference off
                body->add_instruction(instruction::Pop(ast_intr.node_info, ast_intr.signature.rets.size())) ||  // popping the node off the stack
                body->add_instruction(instruction::Ret(ast_intr.node_info)) ||
                bc->add_block(intr_name, *body);
        }

        bool codegen_attr(const std::string& name, const CodeModule::Attr& attr, std::vector<std::string>& ns)
        {
            bool ret = false;
            switch (attr.index())
            {
            case CodeModule::AttrType::NODE:
                for (const auto& [name, intrs] : std::get<CodeModule::Node>(attr).intrs)
                    for (const AstBlock* intr : intrs)
                        ret = codegen_intr(name, *intr, ns) || ret;
                for (const auto& [node_name, node_attrs] : std::get<CodeModule::Node>(attr).attrs)
                {
                    ns.push_back(node_name);
                    for (const auto& node_attr : node_attrs)
                        ret = codegen_attr(node_name, node_attr, ns) || ret;
                    ns.pop_back();
                }
                return ret;
            case CodeModule::AttrType::STRUCT:
                return codegen_struct(name, *std::get<const AstStruct*>(attr), ns);
            case CodeModule::AttrType::FUNC:
                return codegen_func(name, *std::get<const AstFn*>(attr), ns);
            case CodeModule::AttrType::DEF:
                return codegen_def(name, *std::get<const AstBlock*>(attr), ns);
            }
            assert(false);
            return true;
        }

        bool codegen_module(ByteCodeModule& inp_bc, ModuleInfo& info, AstModule& ast, const std::vector<std::string>& imp_dirs)
        {
            bc = &inp_bc;
            mod = &info.mod;
            info.node_info = { ast.fname, 0, 0, 0, 0 };

            // Resolving imports to build a CodeModule object
            std::vector<std::string> visited = { ast.fname };
            if (CodeModule::create(info.mod, ast, imp_dirs, visited))
                return error::compiler(AstNodeInfo(ast.fname, 0, 0, 0, 0), "Unable to generate a code module");

            bool ret = false;
            std::vector<std::string> ns;
            for (auto& [name, intrs] : mod->root.intrs)
                for (const AstBlock* intr : intrs)
                    ret = codegen_intr(name, *intr, ns) || ret;
            for (const auto& [name, attrs] : mod->root.attrs)
                for (const auto& attr : attrs)
                    ret = codegen_attr(name, attr, ns) || ret;
            return ret;
        }
    }
}
