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

namespace nn
{
    namespace lang
    {
        // Module context

        static CodeModule* mod = nullptr;
        static ByteCodeModule* bc = nullptr;
        static std::vector<std::string> cg_ns;  // the current namespace

        // Block context

        static TypeManager* type_manager = nullptr;
        static ByteCodeBody* body = nullptr;
        // Set by codegen_line_while, codegen_line_for.  Read by codegen_line_goto
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
                Arg nd;
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
                dst.inits[init.name].push_back(&init);
            return false;
        }

        bool CodeModule::merge_ast(AstModule& ast)
        {
            return merge_node(root, ast);
        }

        bool CodeModule::create(CodeModule& mod, const AstModule& ast, const std::vector<std::string>& imp_dirs, std::vector<std::string> visited)
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
                nodes   .size() == 0 &&
                structs .size() == 0 &&
                fns     .size() == 0 &&
                defs    .size() == 0 &&
                intrs   .size() == 0 &&
                inits   .size() == 0;
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
                        result.nodes.push_back(std::get<Node>(attr));
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
            return result.empty();
        }

        TypeRef::TypeRef(size_t ptr) : ptr(ptr) {}

        TypeRef::operator bool() const { return ptr; }
        TypeInfo* TypeRef::operator->() { return type_manager->get(ptr); }
        const TypeInfo* TypeRef::operator->() const { return type_manager->get(ptr); }
        TypeInfo& TypeRef::operator*() { return *type_manager->get(ptr); }
        const TypeInfo& TypeRef::operator*() const { return *type_manager->get(ptr); }

        const TypeRef TypeInfo::null = TypeRef();

        TypeInfo::TypeInfo() {}

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
                break;
            case TypeInfo::Type::NODE:
            case TypeInfo::Type::BLOCK:
                // TODO: implement this
            case TypeInfo::Type::EDGE:
            case TypeInfo::Type::TENSOR:
                type_tensor.~TypeInfoTensor();
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
            return false;
        }

        bool TypeInfo::check_xint() const
        {
            return false;
        }

        bool TypeInfo::check_xflt() const
        {
            return false;
        }

        std::string TypeInfo::encode() const
        {
            return "";
        }

        std::string TypeInfo::to_string() const
        {
            return "";
        }

        bool TypeInfo::to_obj(const AstNodeInfo& node_info, TypeRef& type) const
        {
            TypeRef tmp;
            switch (type->ty)
            {
            case TypeInfo::Type::FTY:
                tmp = type_manager->create_fty(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                    size_t addr;
                    return
                        bc->add_type_fty(addr) ||
                        body->add_instruction(instruction::New(node_info, addr));
                }, tmp);
                break;
            case TypeInfo::Type::BOOL:
                tmp = type_manager->create_bool(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                    size_t addr;
                    return
                        bc->add_type_bool(addr) ||
                        body->add_instruction(instruction::New(node_info, addr));
                }, tmp);
                break;
            case TypeInfo::Type::INT:
                tmp = type_manager->create_bool(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                    size_t addr;
                    return
                        bc->add_type_bool(addr) ||
                        body->add_instruction(instruction::New(node_info, addr));
                }, tmp);
                break;
            case TypeInfo::Type::FLOAT:
                tmp = type_manager->create_bool(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                    size_t addr;
                    return
                        bc->add_type_bool(addr) ||
                        body->add_instruction(instruction::New(node_info, addr));
                }, tmp);
                break;
            case TypeInfo::Type::STR:
                tmp = type_manager->create_bool(TypeInfo::Category::VIRTUAL, nullptr);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info](Scope& scope) -> bool {
                    size_t addr;
                    return
                        bc->add_type_str(addr) ||
                        body->add_instruction(instruction::New(node_info, addr));
                }, tmp);
                break;
            case TypeInfo::Type::ARRAY:
                if (type->type_array.elem->to_obj(node_info, tmp))
                    return true;
                tmp = type_manager->create_array(TypeInfo::Category::VIRTUAL, nullptr, tmp);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info, tmp](Scope& scope) -> bool {
                        return
                            tmp->codegen(scope) ||
                            body->add_instruction(instruction::Arr(node_info));
                    }, tmp);
                break;
            case TypeInfo::Type::TUPLE:
            {
                std::vector<TypeRef> elem_types;
                for (TypeRef& elem : type->type_tuple.elems)
                {
                    TypeRef elem_type;
                    if (elem->to_obj(node_info, elem_type))
                        return true;
                    elem_types.push_back(elem_type);
                }
                tmp = type_manager->create_tuple(TypeInfo::Category::VIRTUAL, nullptr, elem_types);
                if (!tmp) return true;
                type = type_manager->create_type(
                    TypeInfo::Category::CONST,
                    [&node_info, elem_types](Scope& scope) -> bool {
                        for (TypeRef elem_type : elem_types)
                            if (elem_type->codegen(scope))
                                return true;
                        return body->add_instruction(instruction::Agg(node_info, elem_types.size()));
                    }, tmp);
                break;
            }
            default:
                return error::compiler(node_info, "Unable to transform the given type into a runtime object");
            }
            return !type;
        }

        TypeInfo* TypeManager::get(size_t ptr)
        {
            return (TypeInfo*)(buf + ptr);
        }

        TypeRef TypeManager::next()
        {
            if (len == bufsz)
            {
                if (bufsz)
                {
                    size_t nsz = 2 * bufsz;
                    void* tmp = realloc(buf, nsz);
                    if (!tmp)
                        return TypeRef(0);
                    buf = (uint8_t*)tmp;
                    bufsz = nsz;
                }
                else
                {
                    size_t nsz = 256 * bufsz;
                    buf = (uint8_t*)malloc(nsz);
                    if (buf)
                        return TypeRef(0);
                    bufsz = nsz;
                    len = sizeof(TypeInfo);  // reserving 0
                }
            }

            TypeRef ret = TypeRef(len);
            len += sizeof(TypeInfo);
            return ret;
        }

        TypeManager::TypeManager()
        {
            buf = (uint8_t*)malloc(1024 * sizeof(uint8_t));
            if (!buf)
                throw std::bad_alloc();
            bufsz = 1024;
        }

        TypeManager::~TypeManager()
        {
            for (size_t i = 0; i < len; i += sizeof(TypeInfo))
                get(i)->~TypeInfo();
            free(buf);
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
            return true;
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

        TypeRef TypeManager::create_edge(TypeInfo::Category cat, CodegenCallback codegen, TypeRef fp, TypeRef shape)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_tensor) TypeInfoTensor{ fp, shape };
            type->ty = TypeInfo::Type::EDGE;
            type->cat = cat;
            type->codegen = codegen;
            return type;
        }

        TypeRef TypeManager::create_tensor(TypeInfo::Category cat, CodegenCallback codegen, TypeRef fp, TypeRef shape)
        {
            TypeRef type = next();
            if (!type) return type;
            new (&type->type_tensor) TypeInfoTensor{ fp, shape };
            type->ty = TypeInfo::Type::TENSOR;
            type->cat = cat;
            type->codegen = codegen;
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
            for (auto& [name, var] : stack_vars)
            {
                assert(var.ptr <= n);
                var.ptr -= n;
            }
            if (parent)
                parent->pop(n);
            return false;
        }

        bool Scope::local_size(size_t& sz, const Scope* scope) const
        {
            if (scope == parent)
            {
                sz = 0;
                return false;
            }
            if (!parent)
                return error::general("Internal error: invalid scope pointer");
            if (parent->local_size(sz, scope))
                return true;
            sz += stack_vars.size();
            return false;
        }

        bool Scope::list_local_vars(std::vector<StackVar>& vars, const Scope* scope)
        {
            size_t sz;
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
            case TypeNode::Type::TENSOR:
                type_tensor.~TensorType();
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
            case TypeNode::Type::TENSOR:
                new (&type_tensor) decltype(type_tensor)(std::move(node.type_tensor));
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
                    cargs.push_back(e.as_type(node_info));
                    if (!cargs.back())
                        return TypeRef();
                }
                return type_manager->create_tuple(TypeInfo::Category::VIRTUAL, nullptr, cargs);
            }
            case TypeNode::Type::TENSOR:
                return type_manager->create_tensor(TypeInfo::Category::VIRTUAL, nullptr, TypeInfo::null, TypeInfo::null);
            default:
                // this code should be unreachable
                error::compiler(node_info, "Internal error: enumeration found outside valid range");
                return TypeRef();
            }
        }

        bool ProcCall::create_arg(Scope& scope, const AstArgDecl& decl, ValNode& node)
        {
            new (&node.val_arg) Arg();
            node.ty = ValNode::Type::ARG_VAL;
            node.node_info = &decl.node_info;
            node.val_arg.name = decl.var_name;
            node.val_arg.type = new TypeNode();
            if (!decl.is_packed)
            {
                // Non-packed arguments can have default values
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *decl.default_expr, node.val_arg.default_type))
                    return true;
                return create_type(*decl.type_expr, *node.val_arg.type);
            }
            // The argument is packed, wrap the type in an array and make sure theres no defaults
            if (decl.default_expr)
                return error::compiler(decl.node_info, "Packed arguments cannot have default values");
            new (&node.val_arg.type->type_array) ArrayType();
            node.val_arg.type->ty = TypeNode::Type::ARRAY;
            node.val_arg.type->node_info = &decl.node_info;
            node.val_arg.type->type_array.carg = new TypeNode{};
            return create_type(*decl.type_expr, *node.val_arg.type->type_array.carg);
        }

        bool ProcCall::create_type(const AstExpr& expr, TypeNode& node)
        {
            Scope scope(nullptr);
            node.node_info = &expr.node_info;
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
                        new (&node.type_array) ArrayType();
                        node.ty = TypeNode::Type::ARRAY;
                        node.type_array.carg = new TypeNode();
                        return create_type(expr.expr_call.args[0], *node.type_array.carg);
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
                    ValNode* pnode = new ValNode{};
                    if (create_value(*expr.expr_call.callee, *pnode))
                    {
                        delete pnode;
                        return true;
                    }

                    std::vector<TypeRef> callee_types;
                    if (pnode->get_type(expr.expr_call.callee->node_info, callee_types))
                        return true;
                    if (callee_types.size() != 1)
                        return error::compiler(expr.expr_call.callee->node_info, "Cargs callee needs to be exactly one value");

                    if (callee_types.front()->ty == TypeInfo::Type::FTY)
                    {
                        // Its a tensor type, this places constraints on the cargs
                        new (&node.type_tensor) TensorType();
                        node.ty = TypeNode::Type::TENSOR;
                        node.type_tensor.fp = pnode;
                        bool has_unpack = false;
                        for (const AstExpr& arg : expr.expr_call.args)
                        {
                            ValNode val_node = ValNode{};
                            if (create_value(arg, val_node))
                                return true;
                            std::vector<TypeRef> val_types;
                            if (val_node.get_type(expr.node_info, val_types))
                                return true;
                            for (TypeRef val_type : val_types)
                                if (val_type->ty != TypeInfo::Type::INT)
                                    return error::compiler(expr.node_info, "Invalid type given as tensor carg, expected int");
                            // Checking for unpacked behaviour
                            if (val_node.ty == ValNode::Type::UNARY_UNPACK)
                            {
                                // Not nessisarilly an array unpack, need to check the type of the value it unpacks
                                std::vector<TypeRef> unpack_rets;
                                if (val_node.val_unary.inp->get_type(expr.node_info, unpack_rets))
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
                            node.type_tensor.shape.push_back(val_node);
                        }
                    }

                    // Its either a struct or an error.  Since structs aren't implemented yet, generate an error regardless
                    return error::compiler(expr.node_info, "Internal error: not implemented");
                }
            case ExprType::KW:
                switch (expr.expr_kw)
                {
                case ExprKW::TYPE:
                    node.ty = TypeNode::Type::TYPE;
                    return false;
                case ExprKW::INIT:
                    node.ty = TypeNode::Type::INIT;
                    return false;
                case ExprKW::FTY:
                    node.ty = TypeNode::Type::FTY;
                    return false;
                case ExprKW::BOOL:
                    node.ty = TypeNode::Type::BOOL;
                    return false;
                case ExprKW::INT:
                    node.ty = TypeNode::Type::INT;
                    return false;
                case ExprKW::FLOAT:
                    node.ty = TypeNode::Type::FLOAT;
                    return false;
                case ExprKW::STR:
                    node.ty = TypeNode::Type::STRING;
                    return false;
                default:
                    return error::compiler(expr.node_info, "Invalid use of keyword '%'", to_string(expr.expr_kw));
                }
            case ExprType::VAR:
                if (carg_nodes.contains(expr.expr_string))
                {
                    std::vector<TypeRef> rets;
                    ValNode& val_node = carg_nodes.at(expr.expr_string);
                    if (val_node.get_type(expr.node_info, rets))
                        return true;
                    if (rets.size() != 1)
                        return error::compiler(expr.node_info, "To use a carg as a type, the carg must be exactly one value");
                    if (rets.front()->ty != TypeInfo::Type::TYPE)
                        return error::compiler(expr.node_info, "Only generic type cargs can be used as a type");
                    new (&node.type_val) ValType();
                    node.ty = TypeNode::Type::GENERIC;
                    node.type_val.val = &val_node;
                    return false;
                }
                // TODO: Implement structs in siguratures
                return error::compiler(expr.node_info, "structs in signatures has not yet been implemented");
            default:
                // everything else is invalid
                return error::compiler(expr.node_info, "Invalid type expression in signature");
            }
        }

        bool ProcCall::create_value(const AstExpr& expr, ValNode& node)
        {
            Scope scope(nullptr);  // dummy scope for stateless codegen
            node.node_info = &expr.node_info;
            switch (expr.ty)
            {
            case ExprType::UNARY_POS:
                new (&node.val_unary) UnaryOp();
                node.ty = ValNode::Type::UNARY_POS;
                node.val_unary.inp = new ValNode{};
                return create_value(*expr.expr_unary.expr, *node.val_unary.inp);
            case ExprType::UNARY_NEG:
                new (&node.val_unary) UnaryOp();
                node.ty = ValNode::Type::UNARY_NEG;
                node.val_unary.inp = new ValNode{};
                return create_value(*expr.expr_unary.expr, *node.val_unary.inp);
            case ExprType::UNARY_NOT:
                new (&node.val_unary) UnaryOp();
                node.ty = ValNode::Type::UNARY_NOT;
                node.val_unary.inp = new ValNode{};
                return create_value(*expr.expr_unary.expr, *node.val_unary.inp);
            case ExprType::BINARY_ADD:
                new (&node.val_binary) BinaryOp();
                node.ty = ValNode::Type::BINARY_ADD;
                node.val_binary.lhs = new ValNode{};
                node.val_binary.rhs = new ValNode{};
                return
                    create_value(*expr.expr_binary.left, *node.val_binary.lhs) ||
                    create_value(*expr.expr_binary.right, *node.val_binary.rhs);
            case ExprType::BINARY_SUB:
                new (&node.val_binary) BinaryOp();
                node.ty = ValNode::Type::BINARY_SUB;
                node.val_binary.lhs = new ValNode{};
                node.val_binary.rhs = new ValNode{};
                return
                    create_value(*expr.expr_binary.left, *node.val_binary.lhs) ||
                    create_value(*expr.expr_binary.right, *node.val_binary.rhs);
            }

            // if all else fails, try codegening the expr and getting a constant value from it
            node.ty = ValNode::Type::CONST_VAL;
            return codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, expr, node.val);
        }

        bool ProcCall::codegen_root_arg(Scope& scope, ValNode& node)
        {
            assert(node.ty == ValNode::Type::ARG_VAL);
            if (node.val_arg.visited)  // Checking if the nodes was already generated and on the stack
                return false;

            TypeRef value_type = node.val;
            if (!value_type)
                value_type = node.val_arg.default_type;
            if (!value_type)
            {
                // I never need to bubble up codegen calls since in ProcCall::codegen_root_arg
                // since its only called on root nodes.  And by definition, root nodes aren't
                // depended on by other nodes, making it impossible for a codegen call to give
                // any root node a value (it can be deduced from the remaining signature)
                return error::compiler(*node.node_info, "Unable to determine a value for arg % in signature", node.val_arg.name);
            }
            node.val = type_manager->duplicate(
                TypeInfo::Category::CONST,
                [value_type, name{ node.val_arg.name }, &node_info{ *node.node_info }](Scope& scope) {
                    return
                        value_type->codegen(scope) ||
                        scope.add(name, value_type, node_info);
                }, value_type);
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
                    body->add_label(*node.node_info, end_label);
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

        bool ProcCall::codegen_type_tensor(Scope& scope, TypeNode& node, TypeRef& type)
        {
            // TODO: figure out how to deal with the a bit better
            if (type->ty != tensor_type)
                return error::compiler(*node.node_info, "Type conflict found during signature deduction");
            if (codegen_value(scope, *node.type_tensor.fp, type->type_tensor.fp))
                return true;
            int unpack_idx = -1;
            for (int i = 0; i < node.type_tensor.shape.size(); i++)
                if (node.type_tensor.shape[i].ty == ValNode::Type::UNARY_UNPACK)
                {
                    if (unpack_idx != -1)
                        return error::compiler(*node.type_tensor.shape[i].node_info, "Found multiple unpacks in tensor's shape");
                    unpack_idx = i;
                }
            if (unpack_idx == -1)
            {
                // Basic case, the shape of the tensor is described explicitly
                std::string end_lbl = label_prefix(*node.node_info) + "_shape_end";
                size_t sz_addr, int_addr, err_addr;
                if (bc->add_obj_int(sz_addr, node.type_tensor.shape.size()) ||
                    bc->add_type_int(int_addr) ||
                    bc->add_obj_str(err_addr, "Tensor rank mismatch found during signature deduction")
                    ) return true;

                if (type->type_tensor.shape->codegen(scope) ||
                    body->add_instruction(instruction::Dup(*node.node_info, 0)) ||
                    scope.push() ||
                    body->add_instruction(instruction::Len(*node.node_info)) ||
                    body->add_instruction(instruction::New(*node.node_info, sz_addr)) ||
                    body->add_instruction(instruction::New(*node.node_info, int_addr)) ||
                    body->add_instruction(instruction::Eq(*node.node_info)) ||
                    body->add_instruction(instruction::Brt(*node.node_info, end_lbl)) ||
                    body->add_instruction(instruction::New(*node.node_info, err_addr)) ||
                    body->add_instruction(instruction::Err(*node.node_info)) ||
                    body->add_label(*node.node_info, end_lbl)
                    ) return true;
                // Confirmed that at runtime the shape matches
                for (int i = 0; i < node.type_tensor.shape.size(); i++)
                {
                    size_t addr;
                    if (bc->add_obj_int(addr, i))
                        return true;
                    TypeRef elem_ty = type_manager->create_int(
                        TypeInfo::Category::CONST,
                        [node_info{ node.node_info }, &type, addr, int_addr](Scope& scope) {
                            return
                                type->type_tensor.shape->codegen(scope) ||
                                body->add_instruction(instruction::New(*node_info, addr)) ||
                                body->add_instruction(instruction::New(*node_info, int_addr)) ||
                                body->add_instruction(instruction::Arr(*node_info)) ||
                                body->add_instruction(instruction::Idx(*node_info));
                        });
                    if (!elem_ty || codegen_value(scope, node.type_tensor.shape[i], elem_ty))
                        return true;
                }
                return false;
            }

            // There was an unpack, so theres just a minimum size constraint placed on the tensor's shape,
            // and the nodes peel elements off both the front and back of the shape
            return error::compiler(*node.node_info, "Internal error: not implemented");
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
            case TypeNode::Type::TENSOR:
                return codegen_type_tensor(scope, node, type);
            default:
                return error::compiler(*node.node_info, "Internal error: enum out of range");
            }
        }

        bool InitCall::init(const AstInit& sig)
        {
            tensor_type = TypeInfo::Type::TENSOR;
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
                ValNode node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                carg_nodes[arg_decl.var_name] = std::move(node);
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
                carg_nodes[name].val = expr;

            return false;
        }

        bool InitCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node.is_root && codegen_root_arg(scope, node))
                    return true;

            // Creating the init object
            size_t addr;
            if (bc->add_obj_str(addr, init_name) ||
                body->add_instruction(instruction::New(*pinfo, addr)) ||
                body->add_instruction(instruction::Ini(*pinfo))
                ) return true;

            // Iterating through the cargs, configuring the init
            for (const auto& [name, node] : carg_nodes)
            {
                // Determining where to get the value of the arg from
                TypeRef val;
                if (node.val)
                    val = node.val;
                else if (node.val_arg.default_type)
                    val = node.val_arg.default_type;
                else
                    return error::compiler(*node.node_info, "Unable to deduce a value for carg '%'", name);
                
                TypeRef type;
                size_t name_addr;
                if (val->to_obj(*node.node_info, type) ||
                    bc->add_obj_str(name_addr, name) ||
                    val->codegen(scope) ||
                    type->codegen(scope) ||
                    body->add_instruction(instruction::New(*node.node_info, name_addr)) ||
                    body->add_instruction(instruction::InCfg(*node.node_info))
                    ) return true;
            }
            
            std::string var_name = generate_var_name();
            TypeRef ret = type_manager->create_init(
                TypeInfo::Category::DEFAULT,
                [var_name, node_info{ this->node_info }](Scope& scope) {
                    Scope::StackVar var;
                    return
                        scope.at(var_name, var) ||
                        body->add_instruction(instruction::Dup(*node_info, var.ptr));
                });
            if (scope.add(var_name, ret, *node_info))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool DefCall::init(const AstBlock& sig)
        {
            tensor_type = TypeInfo::Type::TENSOR;
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;
            cg_ns = sig_ns;

            // Building the nodes for the cargs
            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);

                ValNode node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;

                if (arg_decl.default_expr)
                {
                    if (arg_decl.is_packed)
                        return error::compiler(arg_decl.node_info, "Packed arguments cannot have default values");
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(init_scope, *arg_decl.default_expr, node.val_arg.default_type))
                        return true;
                }
                node.val_arg.name = arg_decl.var_name;
                node.node_info = &arg_decl.node_info;
                carg_nodes[arg_decl.var_name] = std::move(node);
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

                ValNode node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                if (arg_decl.default_expr)
                    return error::compiler(arg_decl.default_expr->node_info, "Default arguments are not allowed in vargs");
                if (node.val_arg.type->ty != TypeNode::Type::TENSOR)
                    return error::compiler(*node.val_arg.type->node_info, "Only tensor types are allowed as vargs in a def");
                node.val_arg.name = arg_decl.var_name;
                node.node_info = &arg_decl.node_info;
                varg_nodes[arg_decl.var_name] = std::move(node);
                varg_stack.push_back(arg_decl.var_name);
            }

            // Building the rets of the def
            for (const AstBlockRet& ret : sig.signature.rets)
            {
                if (ret_nodes.contains(ret.ret_name))
                    return error::compiler(ret.node_info, "Naming conflict for return value '%'", ret.ret_name);

                TypeNode node;
                if (create_type(*ret.type_expr, node))
                    return true;
                if (node.ty != TypeNode::Type::TENSOR)
                    return error::compiler(*node.node_info, "Return types from a def must be tensors");
                ret_nodes[ret.ret_name] = std::move(node);
                ret_stack.push_back(ret.ret_name);
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
                carg_nodes[name].val = expr;

            // Putting the varg exprs into the nodes
            for (size_t i = 0; i < vargs.size(); i++)
                varg_nodes[varg_stack[i]].val = vargs[i];
            return false;
        }

        bool DefCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // codegening all the nodes
            for (auto& [name, node] : carg_nodes)
                if (node.is_root && codegen_root_arg(scope, node))
                    return true;
            for (auto& [name, node] : varg_nodes)
                if (node.is_root && codegen_root_arg(scope, node))
                    return true;

        }

        bool FnCall::init(const AstFn& sig)
        {
            tensor_type = TypeInfo::Type::TENSOR;
            node_info = &sig.node_info;

            std::vector<std::string> old_ns = cg_ns;  // copying out the old namespace
            cg_ns = sig_ns;  // replacing it with the namespace that the sig was defined in

            // Building the nodes for the cargs
            Scope init_scope{ nullptr };
            for (const auto& arg_decl : sig.signature.cargs)
            {
                if (carg_nodes.contains(arg_decl.var_name))
                    return error::compiler(arg_decl.node_info, "Naming conflict for argument '%' in cargs", arg_decl.var_name);

                ValNode node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;

                if (arg_decl.default_expr)
                {
                    if (arg_decl.is_packed)
                        return error::compiler(arg_decl.node_info, "Packed arguments cannot have default values");
                    if (codegen_expr_single_ret<TypeInfo::NonVirtual>(init_scope, *arg_decl.default_expr, node.val_arg.default_type))
                        return true;
                }
                node.val_arg.name = arg_decl.var_name;
                node.node_info = &arg_decl.node_info;
                carg_nodes[arg_decl.var_name] = std::move(node);
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

                ValNode node;
                if (create_arg(init_scope, arg_decl, node))
                    return true;
                if (arg_decl.default_expr)
                    return error::compiler(arg_decl.default_expr->node_info, "Default arguments are not allowed in vargs");
                node.val_arg.name = arg_decl.var_name;
                node.node_info = &arg_decl.node_info;
                varg_nodes[arg_decl.var_name] = std::move(node);
                varg_stack.push_back(arg_decl.var_name);
            }

            // TODO: handle return values

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
                carg_nodes[name].val = expr;

            // Putting the varg exprs into the nodes
            for (size_t i = 0; i < vargs.size(); i++)
                varg_nodes[varg_stack[i]].val = vargs[i];
            return false;
        }

        bool FnCall::codegen(Scope& scope, std::vector<TypeRef>& rets)
        {
            // Doing the argument deduction from the root nodes.
            // This ensures that I will never need to bubble up codegen calls on null value_exprs.
            // But I still need to keep track of the order to arrange the stack before the call.
            for (auto& [name, node] : varg_nodes)
                if (node.is_root && codegen_root_arg(scope, node))
                    return true;
            for (auto& [name, node] : carg_nodes)
                if (node.is_root && codegen_root_arg(scope, node))
                    return true;

            // creating a reverse mapping from arg_type names to stack position
            std::map<std::string, size_t> arg_positions;
            for (size_t i = 0; i < stack_args.size(); i++)
                arg_positions[stack_args[i]] = stack_args.size() - i - 1;

            // dupping the cargs generated from deduction onto the top of the stack in call order
            for (size_t i = 0; i < carg_stack.size(); i++)
            {
                const std::string& name = carg_stack[i];
                if (body->add_instruction(instruction::Dup(*carg_nodes.at(name).node_info, arg_positions.at(name) + i)))
                    return true;
            }
            // dupping the vargs generated from deduction onto the top of the stack in call order
            for (size_t i = 0; i < varg_stack.size(); i++)
            {
                const std::string& name = varg_stack[i];
                if (body->add_instruction(instruction::Dup(*varg_nodes.at(name).node_info, arg_positions.at(name) + i)))
                    return true;
            }

            // popping the (now) duplicate values generated during deduction
            for (size_t i = 0; i < stack_args.size(); i++)
            {
                const std::string& name = stack_args[stack_args.size() - i - 1];  // popping from most recent
                // Figuring out if the argument was a carg or varg
                const AstNodeInfo* node_info = nullptr;
                if (carg_nodes.contains(name))
                {
                    assert(!varg_nodes.contains(name));
                    node_info = carg_nodes.at(name).node_info;
                }
                else
                {
                    assert(varg_nodes.contains(name));
                    node_info = varg_nodes.at(name).node_info;
                }

                if (body->add_instruction(instruction::Pop(*node_info, stack_args.size())))
                    return true;
            }

            // TODO: handle return values

            return false;
        }

        bool ModuleInfo::entry_setup(std::string& ep_name, ByteCodeModule& bc, const std::string& name, const std::map<std::string, std::pair<Obj, TypeInfo>>& cargs) const
        {
            if (!entry_points.contains(name))
                return error::general("Unable to find any entry points with name '%'", name);

            std::vector<EntryPoint*> matches;
            for (auto& ep : entry_points.at(name))
            {
                bool match = true;
                // Checking to make sure that all the given cargs match and the types match
                for (const auto& [name, data] : cargs)
                {

                }
                ep.sig->cargs;
            }
            return error::general("not implemented");
        }

        std::string label_prefix(const AstNodeInfo& type)
        {
            char buf[64];
            sprintf(buf, "l%uc%u_", type.line_start, type.col_start);
            return buf;
        }

        bool arg_type(TypeRef& type, const Scope& scope, const AstArgDecl& arg)
        {
            TypeRef explicit_type;
            if (arg.type_expr)
            {
                // TODO: determine the type from arg_type.type_expr
                return error::compiler(arg.node_info, "Internal error: not implemented");
            }

            TypeRef default_type;
            if (arg.default_expr)
            {
                // TODO: determine the type of arg_type.default_expr
                return error::compiler(arg.node_info, "Internal error: not implemented");
            }

            if (explicit_type->ty == TypeInfo::Type::INVALID)
            {
                if (default_type->ty == TypeInfo::Type::INVALID)
                    return error::compiler(arg.node_info, "Missing both the type expression and the default expression");
                // only the default type was specified
                type = default_type;
                return false;
            }
            if (default_type->ty == TypeInfo::Type::INVALID)
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
                                codegen_expr_attempt_implicit(scope, it->node_info, var.type, cargs.at(it->var_name))
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
                                // Finished eating all the rets from the passed value, move onto the next one.
                                // continue might not be right here...
                                continue;
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
                                if (codegen_expr_attempt_implicit(scope, arg_it->node_info, var.type, *rets_it))
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
                                        if (codegen_expr_attempt_implicit(scope, arg_it->node_info, var.type, *rets_it))
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
            return true;
        }

        bool match_def_sig(Scope& scope, Scope& sig_scope, const AstBlockSig& def,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs)
        {
            // If the cargs don't fit, you must acquit
            if (match_carg_sig(scope, sig_scope, def.cargs, param_cargs, cargs))
                return true;
            
            for (const AstArgDecl& varg : def.vargs)
                if (varg.is_packed)
                    return error::compiler(varg.node_info, "Internal error: packed vargs has not been implemented");

            if (def.vargs.size() == vargs.size())
            {
                vargs = param_vargs;
                return false;
            }
            return true;
        }

        bool match_intr_sig(Scope& scope, Scope& sig_scope, const AstBlockSig& intr,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs)
        {
            if (match_carg_sig(scope, sig_scope, intr.cargs, param_cargs, cargs))
                return true;

            for (const AstArgDecl& varg : intr.vargs)
                if (varg.is_packed)
                    return error::compiler(varg.node_info, "Internal error: packed vargs has not been implemented");

            if (intr.vargs.size() == vargs.size())
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

            return error::compiler(fn.node_info, "Internal error: not implemented");
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
            CodegenCallback cb = [&expr](Scope& scope)
            {
                size_t addr;
                return
                    bc->add_obj_bool(addr, expr.expr_bool) ||
                    body->add_instruction(instruction::New(expr.node_info, addr)) ||
                    scope.push();
            };
            TypeRef ret = type_manager->create_bool(TypeInfo::Category::CONST, std::move(cb));
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_int(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            CodegenCallback cb = [&expr](Scope& scope)
            {
                size_t addr;
                return
                    bc->add_obj_int(addr, expr.expr_int) ||
                    body->add_instruction(instruction::New(expr.node_info, addr)) ||
                    scope.push();
            };
            TypeRef ret = type_manager->create_int(TypeInfo::Category::CONST, std::move(cb));
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_float(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            CodegenCallback cb = [&expr](Scope& scope)
            {
                size_t addr;
                return
                    bc->add_obj_float(addr, expr.expr_float) ||
                    body->add_instruction(instruction::New(expr.node_info, addr)) ||
                    scope.push();
            };
            TypeRef ret = type_manager->create_float(TypeInfo::Category::CONST, std::move(cb));
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_string(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            CodegenCallback cb = [&expr](Scope& scope)
            {
                size_t addr;
                return
                    bc->add_obj_str(addr, expr.expr_string) ||
                    body->add_instruction(instruction::New(expr.node_info, addr)) ||
                    scope.push();
            };
            TypeRef ret = type_manager->create_string(TypeInfo::Category::CONST, std::move(cb));
            if (!ret)
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_array(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            if (expr.expr_agg.elems.size() == 0)
            {
                CodegenCallback cb = [&expr](Scope& scope)
                {
                    return
                        body->add_instruction(instruction::Agg(expr.node_info, 0)) ||
                        scope.push();
                };
                TypeRef ret = type_manager->create_array(TypeInfo::Category::CONST, cb, type_manager->create_placeholder());
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
                scope.push(elem_types.size() - sz);
                sz = elem_types.size();
            }
            return error::compiler(expr.node_info, "Internal error: not implemented");

            // TODO: Find a common element expression type between all of the elements and build type
            return body->add_instruction(instruction::Agg(expr.node_info, sz));
        }

        bool codegen_expr_tuple(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> elem_types;
            for (const auto& elem_expr : expr.expr_agg.elems)
                if (codegen_expr(scope, elem_expr, elem_types))
                    return true;
            CodegenCallback cb = [&expr, elem_types](Scope& scope)
            {
                for (const auto& type : elem_types)
                    type->codegen(scope);
                return
                    body->add_instruction(instruction::Agg(expr.node_info, elem_types.size())) ||
                    scope.pop(elem_types.size() - 1);  // -1 for the added tuple
            };
            TypeRef ret = type_manager->create_tuple(TypeInfo::Category::CONST, std::move(cb), elem_types);
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

        constexpr char binop_errmsg[] = "The % hand side of % operation must be provided with a single value, recieved '%' values";

        bool codegen_expr_attempt_implicit(Scope& scope, const AstNodeInfo& node_info, TypeRef& base, TypeRef& other)
        {
            constexpr char impcast_errmsg[] = "Type '%' cannot be implicitly cast to type '%'";
            CodegenCallback cb;
            switch (base->ty)
            {
            case TypeInfo::Type::INT:
                if (!other->check_xint())
                    return error::compiler(node_info, impcast_errmsg, other->to_string(), "int");
                TypeRef oty;
                if (other->to_obj(node_info, oty))  // this should succeed cause of check_x*()
                    return true;
                cb = [&node_info, oty, codegen{ other->codegen }](Scope& scope)
                {
                    return
                        codegen(scope) ||  // need to copy the codegen function cause it'll get overwritten
                        oty->codegen(scope) ||
                        body->add_instruction(instruction::XInt(node_info));
                };
                break;
            case TypeInfo::Type::FLOAT:
                if (!other->check_xflt())
                    return error::compiler(node_info, impcast_errmsg, other->to_string(), "float");
                TypeRef oty;
                if (other->to_obj(node_info, oty))  // this should succeed cause of check_x*()
                    return true;
                cb = [&node_info, oty, codegen{ other->codegen }](Scope& scope)
                {
                    return
                        codegen(scope) ||  // need to copy the codegen function cause it'll get overwritten
                        oty->codegen(scope) ||
                        body->add_instruction(instruction::XFlt(node_info));
                };
                break;
            case TypeInfo::Type::STR:
                if (!other->check_xstr())
                    return error::compiler(node_info, impcast_errmsg, other->to_string(), "str");
                TypeRef oty;
                if (other->to_obj(node_info, oty))  // this should succeed cause of check_x*()
                    return true;
                cb = [&node_info, oty, codegen{ other->codegen }](Scope& scope)
                {
                    return
                        codegen(scope) ||  // need to copy the codegen function cause it'll get overwritten
                        oty->codegen(scope) ||
                        body->add_instruction(instruction::XStr(node_info));
                };
                break;
            case TypeInfo::Type::ARRAY:
            case TypeInfo::Type::TUPLE:
                // TODO: add recursive implicit casting
                return error::compiler(node_info, "implicit type casting of recursive types has not been implemented");
            default:
                return error::compiler(node_info, impcast_errmsg, other->to_string(), base->to_string());
            }
            other->codegen = cb;
            return false;
        }
        
        template<class allowed>
        bool codegen_expr_single_ret(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            std::vector<TypeRef> rets;
            if (codegen_expr(scope, expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(expr.node_info, "Expected a single value, recieved % values", rets.size() - prev_sz);
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

        bool codegen_expr_unpack(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            std::vector<TypeRef> args;
            if (codegen_expr_multi_ret<TypeInfo::NonVirtual>(scope, *expr.expr_unary.expr, args))
                return true;
            if (args.size() >= 2 == args.size() == 0)
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

        template<class allowed, bool(*check)(TypeInfo&), const char* op_noun, const char* op_verb, class Node>
        bool codegen_expr_binop_same(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<allowed>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", op_noun);
            if (codegen_expr_single_ret<allowed>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", op_noun);

            if (!check(*lhs_type))
                return error::compiler(expr.node_info, "Unable to % values with type '%'", op_verb, lhs_type->to_string());
            if (codegen_expr_attempt_implicit(scope, expr.node_info, lhs_type, rhs_type))  // this will update rhs_type if nessisary
                return true;

            TypeRef ty;
            if (lhs_type->to_obj(expr.node_info, ty))
                return true;

            CodegenCallback cb = [ty, lcb{ lhs_type->codegen }, rcb{ rhs_type->codegen }](Scope& scope)
            {
                return
                    lcb(scope) ||  // codegen the lhs
                    rcb(scope) ||  // codegen the rhs
                    ty->codegen(scope) ||
                    body->add_instruction(Node(expr.node_info)) ||
                    scope.pop(1);
            };

            ret = type_manager->duplicate(TypeInfo::Category::CONST, cb, lhs_type);
            return !ret;
        }

        template<class lhs_allowed, class rhs_allowed, bool(*check)(TypeInfo&), const char* op_noun, const char* op_verb, class Node>
        bool codegen_expr_binop_iop(Scope& scope, const AstExpr& expr)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<lhs_allowed>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", op_noun);
            if (codegen_expr_single_ret<rhs_allowed>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", op_noun);

            if (!check(*lhs_type))
                return error::compiler(expr.node_info, "Unable to % values with type '%'", op_verb, lhs_type->to_string());
            if (codegen_expr_attempt_implicit(scope, expr.node_info, lhs_type, rhs_type))
                return true;

            TypeRef ty;
            return
                lhs_type->to_obj(expr.node_info, ty) ||
                lhs_type->codegen(scope) ||  // codegen the lhs
                rhs_type->codegen(scope) ||  // codegen the rhs
                ty->codegen(scope) ||
                body->add_instruction(Node(expr.node_info)) ||
                scope.pop(2);
        }

        template<class allowed, bool(*check)(TypeInfo&), const char* op_noun, const char* op_verb, class Node>
        bool codegen_expr_binop_bool(Scope& scope, const AstExpr& expr, TypeRef& ret)
        {
            TypeRef lhs_type, rhs_type;
            if (codegen_expr_single_ret<allowed>(scope, *expr.expr_binary.left, lhs_type))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", op_noun);
            if (codegen_expr_single_ret<allowed>(scope, *expr.expr_binary.right, rhs_type))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", op_noun);

            if (!check(*lhs_type))
                return error::compiler(expr.node_info, "Unable to % values with type '%'", op_verb, lhs_type->to_string());
            if (codegen_expr_attempt_implicit(scope, expr.node_info, lhs_type, rhs_type))
                return true;

            TypeRef ty;
            if (lhs_type->to_obj(expr.node_info, ty))
                return true;

            CodegenCallback cb = [ty, lcb{ lhs_type->codegen }, rcb{ rhs_type->codegen }](Scope& scope)
            {
                return
                    lcb(scope) ||  // codegen the lhs
                    rcb(scope) ||  // codegen the rhs
                    ty->codegen(scope) ||
                    body->add_instruction(Node(expr.node_info)) ||
                    scope.pop(1);
            };

            ret = type_manager->create_bool(TypeInfo::Category::CONST, cb);
            return !ret;
        }

        bool codegen_expr_add(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "addition";
            constexpr static char op_verb[] = "add";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_add(); };
            TypeRef ret;
            if (codegen_expr_binop_same<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Add>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_sub(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "subtraction";
            constexpr static char op_verb[] = "subtract";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_sub(); };
            TypeRef ret;
            if (codegen_expr_binop_same<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Sub>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_mul(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "multiplication";
            constexpr static char op_verb[] = "multiply";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_mul(); };
            TypeRef ret;
            if (codegen_expr_binop_same<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Mul>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_div(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "division";
            constexpr static char op_verb[] = "divide";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_div(); };
            TypeRef ret;
            if (codegen_expr_binop_same<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Div>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_mod(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "modulus";
            constexpr static char op_verb[] = "modulus";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_mod(); };
            TypeRef ret;
            if (codegen_expr_binop_same<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Mod>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_iadd(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "addition assignment";
            constexpr static char op_verb[] = "add";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_add(); };
            return codegen_expr_binop_iop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IAdd>(scope, expr);
        }

        bool codegen_expr_isub(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "subtraction assignment";
            constexpr static char op_verb[] = "subtract";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_sub(); };
            return codegen_expr_binop_iop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::ISub>(scope, expr);
        }

        bool codegen_expr_imul(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "multiplication assignment";
            constexpr static char op_verb[] = "multiply";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_mul(); };
            return codegen_expr_binop_iop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IMul>(scope, expr);
        }

        bool codegen_expr_idiv(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "division assignment";
            constexpr static char op_verb[] = "divide";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_div(); };
            return codegen_expr_binop_iop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IDiv>(scope, expr);
        }

        bool codegen_expr_imod(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "modulus assignment";
            constexpr static char op_verb[] = "modulus";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_mod(); };
            return codegen_expr_binop_iop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IMod>(scope, expr);
        }

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
                TypeRef type;
                if (codegen_expr_single_ret<TypeInfo::NonVirtual>(scope, *rhs, type))
                    return true;
                // doing the codegen for the right hand side
                type->codegen(scope);

                // Overwriting the codegen with a dup instruction
                CodegenCallback cb = [lhs](Scope& scope)
                {
                    size_t addr;
                    Scope::StackVar stack_var;
                    return
                        scope.at(lhs->expr_string, stack_var) ||
                        body->add_instruction(instruction::Dup(lhs->node_info, stack_var.ptr));
                };
                type->codegen = cb;
                // Overwriting the type category based on the modifiers
                // the non-virtual condition is handled by codegen_expr_single_ret<TypeInfo::NonVirtual>
                if (is_const)
                    type->cat = TypeInfo::Category::CONST;
                else if (is_ref)
                {
                    if (!type->in_category<TypeInfo::Mutable>())
                        return error::compiler(expr.node_info, "Unable to take a reference of an immutable object");
                    type->cat = TypeInfo::Category::REF;
                }
                else
                {
                    // objects that are handled by the graph builder
                    // don't have runtime type objects, and can't be copied
                    if (type->check_cpy())
                    {
                        // copying the object during assignment
                        TypeRef ty;
                        if (type->to_obj(expr.node_info, ty) ||
                            ty->codegen(scope) ||
                            body->add_instruction(instruction::Cpy(expr.node_info))
                            ) return true;
                    }
                    type->cat = TypeInfo::Category::DEFAULT;
                }
                return scope.add(lhs->expr_string, type, expr.node_info);
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
                    return
                        lhs_type->codegen(scope) ||
                        rhs_type->codegen(scope) ||
                        body->add_instruction(instruction::Mrg(expr.node_info)) ||
                        scope.pop(2);
                }
                
                return
                    codegen_expr_attempt_implicit(scope, expr.node_info, lhs_type, rhs_type) ||
                    lhs_type->codegen(scope) ||
                    rhs_type->codegen(scope) ||
                    lhs_type->to_obj(expr.node_info, ty) ||
                    ty->codegen(scope) ||
                    body->add_instruction(instruction::Set(expr.node_info)) ||
                    scope.pop(2);
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

        bool codegen_expr_eq(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "equality";
            constexpr static char op_verb[] = "check the equality of";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_eq(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Eq>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_ne(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "inequality";
            constexpr static char op_verb[] = "check the inequality of";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_ne(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Ne>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_gt(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "greater than";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_gt(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Gt>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_lt(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "less than";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_lt(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Lt>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_ge(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "greater than or equal to";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_ge(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Ge>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_le(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            constexpr static char op_noun[] = "less than or equal to";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& type) -> bool { return type.check_le(); };
            TypeRef ret;
            if (codegen_expr_binop_bool<TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Gt>(scope, expr, ret))
                return true;
            rets.push_back(ret);
            return false;
        }

        bool codegen_expr_idx(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
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
                for (const auto& node : lookup.nodes)
                {
                    if (node.attrs.contains(expr.expr_name.val))
                    {
                        for (const auto& attr : node.attrs.at(expr.expr_name.val))
                        {
                            switch (attr.index())
                            {
                            case CodeModule::AttrType::NODE:
                                result.nodes.push_back(std::get<CodeModule::Node>(attr));
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
                    if (node.intrs.contains(expr.expr_name.val))
                        result.intrs = node.intrs.at(expr.expr_name.val);
                    if (node.inits.contains(expr.expr_name.val))
                        result.inits = node.inits.at(expr.expr_name.val);
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
                return error::compiler(expr.node_info, "Variable declarations require a type");
            TypeRef type = decl_type->type_type.base;
            if (type->cat != TypeInfo::Category::DEFAULT)
                return error::compiler(expr.node_info, "Invalid type category for variable declaration");
            if (type->codegen(scope) ||
                scope.push()
                ) return true;
            type->codegen = [&expr](Scope& scope) {
                Scope::StackVar var;
                if (scope.at(expr.expr_name.val, var))
                    return error::compiler(expr.node_info, "Undefined variable %", expr.expr_name.val);
                return
                    body->add_instruction(instruction::Dup(expr.node_info, var.ptr)) ||
                    scope.push();
            };
            return scope.add(expr.expr_name.val, type, expr.node_info);
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
                    TypeRef type = type_manager->create_array(
                        TypeInfo::Category::CONST,
                        [&arg_type, &expr](Scope& scope) {
                            return
                                arg_type->codegen(scope) ||
                                body->add_instruction(instruction::Arr(expr.node_info));
                        },
                        arg_type);
                    if (!type)
                        return true;
                    rets.push_back(type);
                    return false;
                }
                case ExprKW::TUPLE:
                {
                    std::vector<TypeRef> arg_types;
                    for (const auto& arg : expr.expr_call.args)
                        if (codegen_expr(scope, arg, arg_types))
                            return true;
                    for (const auto& arg_type : arg_types)
                    {
                        if (arg_type->ty != TypeInfo::Type::TYPE)
                            return error::compiler(expr.node_info, "A tuple type must be constructed from other types");
                        if (arg_type->cat == TypeInfo::Category::VIRTUAL)
                            return error::compiler(expr.node_info, "Tuple cargs must be non-virtual");
                    }
                    TypeRef type = type_manager->create_tuple(
                        TypeInfo::Category::CONST,
                        [&arg_types, &expr](Scope& scope) {
                            for (const auto& arg_type : arg_types)
                                if (arg_type->codegen(scope))
                                    return true;
                            if (body->add_instruction(instruction::Aty(expr.node_info, arg_types.size())))
                                return true;
                            if (arg_types.size() == 0)
                                return scope.push();
                            if (arg_types.size() == 1)
                                return false;
                            return scope.pop(arg_types.size() - 1);
                        },
                        arg_types);
                    if (!type)
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
                    return error::compiler(expr.node_info, "Interal error: not implemented");
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
                size_t addr;
                if (bc->add_type_int(addr))
                    return true;
                TypeRef tmp;
                tmp = type_manager->create_int(TypeInfo::Category::VIRTUAL, nullptr);
                tmp = type_manager->create_array(
                    TypeInfo::Category::CONST,
                    [&expr, args, addr](Scope& scope) {
                        bool init_stack = true;
                        size_t adjacents = 0;
                        for (TypeRef arg : args)
                        {
                            if (arg->ty == TypeInfo::Type::INT)
                            {
                                if (arg->codegen(scope))
                                    return true;
                                adjacents += 1;
                            }
                            else
                            {
                                assert(arg->ty == TypeInfo::Type::UNPACK);
                                if (adjacents == 0)
                                {
                                    if (arg->codegen(scope))
                                        return true;
                                    if (init_stack)
                                        init_stack = false;
                                    else
                                    {
                                        if (body->add_instruction(instruction::New(expr.node_info, addr)) ||
                                            body->add_instruction(instruction::Arr(expr.node_info)) ||
                                            body->add_instruction(instruction::Add(expr.node_info))
                                            ) return true;
                                    }
                                }
                                else
                                {
                                    if (body->add_instruction(instruction::Agg(expr.node_info, adjacents)))
                                        return true;
                                    if (init_stack)
                                        init_stack = false;
                                    else
                                    {
                                        if (body->add_instruction(instruction::New(expr.node_info, addr)) ||
                                            body->add_instruction(instruction::Arr(expr.node_info)) ||
                                            body->add_instruction(instruction::Add(expr.node_info))
                                            ) return true;
                                    }
                                    if (arg->codegen(scope) ||
                                        body->add_instruction(instruction::New(expr.node_info, addr)) ||
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||
                                        body->add_instruction(instruction::Add(expr.node_info))
                                        ) return true;
                                    adjacents = 0;
                                }
                            }
                        }
                        if (adjacents == 0 && init_stack)
                        {
                            if (body->add_instruction(instruction::Agg(expr.node_info, 0)))
                                return true;
                        }
                        else if (adjacents != 0)
                        {
                            if (body->add_instruction(instruction::Agg(expr.node_info, adjacents)))
                                return true;
                            if (!init_stack)
                            {
                                if (body->add_instruction(instruction::New(expr.node_info, addr)) ||
                                    body->add_instruction(instruction::Arr(expr.node_info)) ||
                                    body->add_instruction(instruction::Add(expr.node_info))
                                    ) return true;
                            }
                        }
                        return false;
                    },
                    tmp);
                tmp = type_manager->create_tensor(
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
                        size_t addr;
                        if (bc->add_obj_int(addr, start_val) ||
                            body->add_instruction(instruction::New(expr.node_info, addr))
                            ) return true;
                        // for arg in args: dup n; dup arg; len; new type int; iadd
                        if (bc->add_type_int(addr))
                            return true;
                        for (size_t i = 0; i < args.size(); i++)
                            if (args[i]->ty == TypeInfo::Type::UNPACK)
                            {
                                if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||
                                    body->add_instruction(instruction::Dup(expr.node_info, args.size() - i + 1)) ||
                                    body->add_instruction(instruction::Len(expr.node_info)) ||
                                    body->add_instruction(instruction::New(expr.node_info, addr)) ||
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
                            if (body->add_instruction(instruction::Dup(expr.node_info, args.size() + 2)) ||  // dup fp
                                body->add_instruction(instruction::Dup(expr.node_info, 2))                   // dup n
                                ) return true;
                            // for arg in args: dup <arg>
                            for (size_t i = 0; i < args.size(); i++)
                                if (body->add_instruction(instruction::Dup(expr.node_info, args.size() + 3)))
                                    return true;

                            // Expanding each of the packed arguments onto the stack
                            for (size_t k = 0; k < args.size(); k++)
                                // TypeInfo::Type::INT will get handled automatically due to the setup
                                if (args[k]->ty == TypeInfo::Type::UNPACK)
                                {
                                    std::string loop_label;
                                    std::string end_label;

                                    // Break condition for the loop over args[k]
                                    if (body->add_instruction(instruction::New(expr.node_info, int0)) ||             // new int 0  # i
                                        body->add_label(expr.node_info, loop_label) ||                               // :loop
                                        body->add_instruction(instruction::Dup(expr.node_info, args.size() - k)) ||  // dup <args[k]>
                                        body->add_instruction(instruction::New(expr.node_info, addr)) ||             // new type int
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||                   // arr
                                        body->add_instruction(instruction::Len(expr.node_info)) ||                   // len
                                        body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                // dup i
                                        body->add_instruction(instruction::New(expr.node_info, addr)) ||             // new type int
                                        body->add_instruction(instruction::Eq(expr.node_info)) ||                    // eq
                                        body->add_instruction(instruction::Brt(expr.node_info, end_label))           // brt end
                                        ) return true;

                                    // Placing the next element from args[k] onto the stack
                                    if (body->add_instruction(instruction::Dup(expr.node_info, args.size() - k)) ||  // dup args[k]
                                        body->add_instruction(instruction::Dup(expr.node_info, 1)) ||                // dup i
                                        body->add_instruction(instruction::New(expr.node_info, addr)) ||             // new type int
                                        body->add_instruction(instruction::Arr(expr.node_info)) ||                   // arr
                                        body->add_instruction(instruction::Idx(expr.node_info))                      // idx
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
                                    if (body->add_instruction(instruction::Dup(expr.node_info, 0)) ||     // dup i
                                        body->add_instruction(instruction::New(expr.node_info, int1)) ||  // new int 1
                                        body->add_instruction(instruction::New(expr.node_info, addr)) ||  // new type int
                                        body->add_instruction(instruction::IAdd(expr.node_info))          // iadd
                                        ) return true;

                                    // Looping back and stack cleanup
                                    if (body->add_instruction(instruction::Jmp(expr.node_info, loop_label)) ||       // jmp loop
                                        body->add_label(expr.node_info, end_label) ||                                // :end
                                        body->add_instruction(instruction::Pop(expr.node_info, args.size() - k)) ||  // pop args[k]
                                        body->add_instruction(instruction::Pop(expr.node_info, 0))                   // pop i
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
                        scope.push();
                        return false;
                    },
                    callee, tmp);
                TypeRef type = type_manager->create_type(TypeInfo::Category::VIRTUAL, nullptr, tmp);
            }
            else
                return error::compiler(expr.node_info, "cargs cannot be applied to the given type");
        }

        bool codegen_expr_vargs(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets)
        {
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
                    if (!match_def_sig(scope, sig_scope, def->signature, *param_cargs, param_vargs, cargs, vargs))
                        def_matches.push_back({ def, std::move(cargs), std::move(vargs) });
                }
                if (def_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to def '%' is ambiguous", callee->type_lookup.name);
                if (def_matches.size() == 1)
                {
                    // Perfect match, return it
                    return error::compiler(expr.node_info, "Interal error: not implemented");
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
                    if (!match_intr_sig(scope, sig_scope, intr->signature, *param_cargs, param_vargs, cargs, vargs))
                        intr_matches.push_back({ intr, std::move(cargs), std::move(vargs) });
                }
                if (intr_matches.size() > 1)
                    return error::compiler(expr.node_info, "Reference to intr '%' is ambiguous", callee->type_lookup.name);
                if (intr_matches.size() == 1)
                {
                    // Perfect match, return it
                    return error::compiler(expr.node_info, "Interal error: not implemented");
                }
            }
            // It doesn't match a def nor an intr, try for a function
            std::vector<std::tuple<const AstFn*, std::map<std::string, TypeRef>, std::vector<TypeRef>>> fn_matches;
            for (const AstFn* fn : callee->type_lookup.lookup.fns)
            {
                std::map<std::string, TypeRef> cargs;
                std::vector<TypeRef> vargs;
                Scope sig_scope{ nullptr };
                if (!match_fn_sig(scope, sig_scope, fn->signature, *param_cargs, param_vargs, cargs, vargs))
                    fn_matches.push_back({ fn, std::move(cargs), std::move(vargs) });
            }
            if (fn_matches.size() > 1)
                return error::compiler(expr.node_info, "Reference to fn '%' is ambiguous", callee->type_lookup.name);
            if (fn_matches.size() == 0)
            {
                // Perfect match, return it
                return error::compiler(expr.node_info, "Internal error: not implemented");
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
                            body->add_instruction(instruction::Ini(expr.node_info));
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
            case ExprType::BINARY_IDX:
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
            if (scope.list_local_vars(vars, scope.parent))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body->add_instruction(instruction::Pop(type, var.ptr)))
                    return true;
            }
            scope.parent->pop(vars.size());
            return false;
        }

        bool codegen_line_break(Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "break statements are not allowed outside a looping structure");

            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, loop_ctx->scope))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body->add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            }
            return body->add_instruction(instruction::Jmp(line.node_info, loop_ctx->break_label));
        }

        bool codegen_line_continue(Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "continue statements are not allowed outside a looping structure");

            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, loop_ctx->scope))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body->add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            }
            return body->add_instruction(instruction::Jmp(line.node_info, loop_ctx->break_label));
        }

        bool codegen_line_export(Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_extern(Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_raise(Scope& scope, const AstLine& line)
        {
            std::vector<TypeRef> rets;
            if (codegen_expr(scope, line.line_func.expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(line.node_info, "A raise expression must resolve to a single string value");
            if (rets[0]->ty != TypeInfo::Type::STR)
                return body->add_instruction(instruction::Err(line.node_info));
            else if (!rets[0]->check_xstr())
                return error::compiler(line.node_info, "A raise expression must resolve to a single string value");
            return
                body->add_instruction(instruction::XStr(line.node_info)) ||
                body->add_instruction(instruction::Err(line.node_info));
        }

        bool codegen_line_print(Scope& scope, const AstLine& line)
        {
            std::vector<TypeRef> rets;
            if (codegen_expr(scope, line.line_func.expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(line.node_info, "A print expression must resolve to a single string value");
            if (rets[0]->ty != TypeInfo::Type::STR)
                return body->add_instruction(instruction::Dsp(line.node_info));
            else if (!rets[0]->check_xstr())
                return error::compiler(line.node_info, "A print expression must resolve to a single string value");
            return
                body->add_instruction(instruction::XStr(line.node_info)) ||
                body->add_instruction(instruction::Dsp(line.node_info));
        }

        bool codegen_line_return(Scope& scope, const AstLine& line)
        {
            // TODO: codegen the return value using the signature to disambiguate tuple return values vs multiple return values
            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, nullptr))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
                if (body->add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            if (ret_label)
                return body->add_instruction(instruction::Jmp(line.node_info, *ret_label));
            return body->add_instruction(instruction::Ret(line.node_info));
        }

        bool codegen_line_branch(Scope& scope, const AstLine& line, const std::string& end_label)
        {
            Scope block_scope{ &scope };
            Scope cond_scope{ &scope };
            std::vector<TypeRef> rets;
            if (codegen_expr(cond_scope, line.line_branch.cond, rets) ||
                codegen_exit(cond_scope, line.node_info)  // leaving the conditional's scope
                ) return true;
            if (rets.size() != 1 || rets[0]->ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a single boolean value");
            std::string false_branch = label_prefix(line.node_info) + "false_branch";
            return
                body->add_instruction(instruction::Brf(line.node_info, false_branch)) ||
                codegen_lines(block_scope, line.line_branch.body) ||
                codegen_exit(block_scope, line.node_info) || // leaving the block's scope
                body->add_instruction(instruction::Jmp(line.node_info, end_label)) ||
                body->add_label(line.node_info, false_branch);
        }

        bool codegen_line_while(Scope& scope, const AstLine& line)
        {
            LoopContext* old_ctx = loop_ctx;
            Scope cond_scope{ &scope };
            Scope block_scope{ &scope };
            std::string loop_start = label_prefix(line.node_info) + "loop_start";
            std::string loop_end = label_prefix(line.node_info) + "loop_end";
            std::vector<TypeRef> rets;
            if (body->add_label(line.node_info, loop_start) ||
                codegen_expr(cond_scope, line.line_branch.cond, rets) ||
                codegen_exit(cond_scope, line.node_info)
                ) return true;
            LoopContext new_ctx = { &block_scope, loop_start, loop_end };
            loop_ctx = &new_ctx;
            if (rets.size() != 1 || rets[0]->ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a single boolean value");
            bool ret =
                body->add_instruction(instruction::Brf(line.node_info, loop_end)) ||
                codegen_lines(block_scope, line.line_branch.body) ||
                body->add_instruction(instruction::Jmp(line.node_info, loop_start)) ||
                body->add_label(line.node_info, loop_end);
            loop_ctx = old_ctx;
            return ret;
        }

        bool codegen_line_for(Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
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
                if (type->codegen(scope) || body->add_instruction(instruction::Pop(line.node_info, 0)))
                    return true;
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
            assert(lines.size() > 0);

            size_t i = 0;
            while (i < lines.size())
            {
                if (lines[i].ty == LineType::IF)
                {
                    std::string end_label = label_prefix(lines[i].node_info) + "branch_end";
                    if (codegen_line_branch(scope, lines[i], end_label))
                        return true;
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
                    }
                    body->add_label(lines[i].node_info, end_label);
                    i++;
                }
                else
                    codegen_line(scope, lines[i]);
            }

            // During proper code execution, this point should be unreachable.
            // If this point is reached during execution, it will fall through to the next function which is undefined
            // and will cause the interpreter to start bahaving unpredictably which would be very difficult to debug.
            // For safety, a raise statement is added manually by the compiler to generate a runtime error
            // if the programmer made a mistake.  If this point was in fact unreachable,
            // the raise will never get run and will not have any consequences (besides moving labels around in the bytecode)
            size_t addr;
            return
                bc->add_obj_str(addr, "Reached the end of the procedure without returning") ||
                body->add_instruction(instruction::New(lines[lines.size() - 1].node_info, addr)) ||
                body->add_instruction(instruction::Err(lines[lines.size() - 1].node_info));
        }

        bool codegen_struct(const std::string& name, const AstStruct& ast_struct, const std::vector<std::string>& ns)
        {
            ByteCodeBody body{ ast_struct.node_info };

            return error::compiler(ast_struct.node_info, "Not implemented");
        }

        bool codegen_func(const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns)
        {
            ByteCodeBody fn_body{ ast_fn.node_info };
            TypeManager fn_type_manager{};
            body = &fn_body;
            type_manager = &fn_type_manager;
            cg_ns = ns;

            Scope scope{ nullptr };
            std::stringstream fn_name;
            fn_name << "fn_";
            for (const auto& ns_name : ns)
                fn_name << ns_name << "_";
            fn_name
                << ast_fn.signature.cargs.size() << "_"
                << ast_fn.signature.vargs.size() << "_";
            for (const auto& arg : ast_fn.signature.cargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info)
                    ) return true;
                fn_name << type->encode() << "_";
            }
            for (const auto& arg : ast_fn.signature.vargs)
            {
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info)
                    ) return true;
                fn_name << type->encode() << "_";
            }
            TypeRef blk_ty = type_manager->create_block(TypeInfo::Category::CONST);
            if (!blk_ty)
                return true;
            if (scope.add("~block", blk_ty, ast_fn.node_info))
                return true;
            fn_name << name;
            return
                codegen_lines(scope, ast_fn.body) ||
                bc->add_block(fn_name.str(), *body);
        }

        bool codegen_def(const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns)
        {
            ByteCodeBody def_body{ ast_def.node_info };
            TypeManager def_type_manager{};
            body = &def_body;
            type_manager = &def_type_manager;
            cg_ns = ns;

            Scope scope{ nullptr };
            std::stringstream def_name;
            def_name << "def" << ns.size() << "_";
            for (const auto& ns_name : ns)
                def_name << ns_name << "_";
            def_name << ast_def.signature.vargs.size() << "_";  // You can only overload defs based on vargs

            // Constructing the scope
            for (const auto& arg : ast_def.signature.cargs)
            {
                // Adding each of the cargs onto the stack
                // At the bytecode level, packed arguments are a single value
                TypeRef type;
                if (arg_type(type, scope, arg) ||
                    scope.add(arg.var_name, type, arg.node_info)
                    ) return true;
            }
            for (const auto& arg : ast_def.signature.vargs)
            {
                // Adding each of the vargs onto the stack
                TypeRef type;
                if (arg_type(type, scope, arg))
                    return true;
                if (type->ty != TypeInfo::Type::TENSOR)
                    return error::compiler(arg.node_info, "def arguments must be tensors");
                if (scope.add(arg.var_name, type, arg.node_info))
                    return true;
            }
            // Adding the block reference onto the stack
            TypeRef blk_ty = type_manager->create_block(TypeInfo::Category::CONST);
            if (!blk_ty)
                return true;
            if (scope.add("~block", blk_ty, ast_def.node_info))
                return true;
            def_name << name;

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
                if (var.type->runtime_obj())
                {
                    size_t addr;
                    TypeRef ty;
                    if (body->add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                        var.type->to_obj(arg.node_info, ty) ||
                        ty->codegen(scope) ||
                        bc->add_obj_str(addr, arg.var_name) ||
                        body->add_instruction(instruction::New(arg.node_info, addr)) ||
                        body->add_instruction(instruction::BkCfg(arg.node_info))
                        ) return true;
                }
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
                return true;
            ret_label = nullptr;

            // Bytecode for the return subroutine

            if (body->add_label(ast_def.node_info, def_ret_label) ||
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

            if (bc->has_proc(def_name.str()))
                return error::compiler(ast_def.node_info, "Unable to overload def blocks by vargs or rets");
            return
                body->add_instruction(instruction::Pop(ast_def.node_info, 0)) ||  // popping the block off for the type
                body->add_instruction(instruction::Ret(ast_def.node_info)) ||
                bc->add_block(def_name.str(), *body);
        }

        bool codegen_intr(const std::string& name, const AstBlock& ast_intr, const std::vector<std::string>& ns)
        {
            cg_ns = ns;
            return error::compiler(ast_intr.node_info, "Not implemented");
        }

        bool codegen_attr(const std::string& name, const CodeModule::Attr& attr, std::vector<std::string>& ns)
        {
            switch (attr.index())
            {
            case CodeModule::AttrType::NODE:
                for (const auto& [node_name, node_attrs] : std::get<CodeModule::Node>(attr).attrs)
                {
                    ns.push_back(node_name);
                    for (const auto& node_attr : node_attrs)
                        if (codegen_attr(node_name, node_attr, ns))
                            return true;
                    ns.pop_back();
                }
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

        bool codegen_module(ByteCodeModule& inp_bc, ModuleInfo& info, const AstModule& ast, const std::vector<std::string>& imp_dirs)
        {
            bc = &inp_bc;
            // Resolving imports to build a CodeModule object
            CodeModule inp_mod;
            std::vector<std::string> visited = { ast.fname };
            if (CodeModule::create(inp_mod, ast, imp_dirs, visited))
                return true;
            mod = &inp_mod;

            std::vector<std::string> ns;
            for (const auto& [name, attrs] : mod->root.attrs)
                for (const auto& attr : attrs)
                    if (codegen_attr(name, attr, ns))
                        return true;
            return false;
        }
    }
}
