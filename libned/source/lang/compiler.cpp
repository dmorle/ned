#include <ned/errors.h>
#include <ned/lang/compiler.h>

#include <map>
#include <unordered_map>
#include <filesystem>
#include <variant>
#include <algorithm>
#include <functional>

namespace fs = std::filesystem;

namespace nn
{
    namespace lang
    {
        template<typename T>
        bool CodeModule::merge_node(Node& dst, T& src)
        {
            for (AstNamespace& ns : src.namespaces)
            {
                Node nd;
                if (merge_node(nd, ns))
                    return true;
                dst.attrs[ns.name].push_back(std::move(nd));
            }
            for (AstStruct& agg : src.structs)
                dst.attrs[agg.signature.name].push_back(std::move(agg));
            for (AstFn& fn : src.funcs)
                dst.attrs[fn.signature.name].push_back(std::move(fn));
            for (AstBlock& def : src.defs)
                dst.attrs[def.signature.name].push_back(std::move(def));
            for (AstBlock& intr : src.intrs)
                dst.intrs[intr.signature.name].push_back(std::move(intr));
            for (AstInit& init : src.inits)
                dst.inits[init.name].push_back(std::move(init));
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
        }

        TypeInfo::TypeInfo() {}

        TypeInfo::TypeInfo(const TypeInfo& type)
        {
            do_copy(type);
        }

        TypeInfo& TypeInfo::operator=(const TypeInfo& type)
        {
            if (&type == this)
                return *this;
            this->~TypeInfo();
            do_copy(type);
            return *this;
        }

        TypeInfo::~TypeInfo()
        {
            switch (ty)
            {
            case TypeInfo::Type::INVALID:
            case TypeInfo::Type::TYPE:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::FWIDTH:
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
            case TypeInfo::Type::NAMESPACE:
                type_namespace.~TypeInfoNamespace();
                break;
            case TypeInfo::Type::STRUCTREF:
                type_struct_ref.~TypeInfoStructRef();
                break;
            case TypeInfo::Type::STRUCT:
                type_struct.~TypeInfoStruct();
                break;
            case TypeInfo::Type::FNREF:
                type_fn_ref.~TypeInfoFnRef();
                break;
            case TypeInfo::Type::FN:
                type_fn.~TypeInfoFn();
                break;
            case TypeInfo::Type::DEFREF:
                type_def_ref.~TypeInfoDefRef();
                break;
            case TypeInfo::Type::DEF:
                type_def.~TypeInfoDef();
                break;
            case TypeInfo::Type::INTRREF:
                type_intr_ref.~TypeInfoIntrRef();
                break;
            case TypeInfo::Type::INTR:
                type_intr.~TypeInfoIntr();
                break;
            case TypeInfo::Type::TENSOR:
            case TypeInfo::Type::EDGE:
                break;
            case TypeInfo::Type::GENERIC:
                type_generic.~TypeInfoGeneric();
                break;
            case TypeInfo::Type::ARRPACK:
                type_arr_pack.~TypeInfoArrPack();
                break;
            case TypeInfo::Type::AGGPACK:
                type_agg_pack.~TypeInfoAggPack();
                break;
            default:
                assert(false);
            }
        }

        void TypeInfo::do_copy(const TypeInfo& type)
        {
            ty = type.ty;
            cat = type.cat;

            switch (ty)
            {
            case TypeInfo::Type::INVALID:
            case TypeInfo::Type::TYPE:
            case TypeInfo::Type::BOOL:
            case TypeInfo::Type::FWIDTH:
            case TypeInfo::Type::INT:
            case TypeInfo::Type::FLOAT:
            case TypeInfo::Type::STR:
                break;
            case TypeInfo::Type::ARRAY:
                new (&type_array) decltype(type_array)(type.type_array);
                break;
            case TypeInfo::Type::TUPLE:
                new (&type_tuple) decltype(type_tuple)(type.type_tuple);
                break;
            case TypeInfo::Type::NAMESPACE:
                new (&type_namespace) decltype(type_namespace)(type.type_namespace);
                break;
            case TypeInfo::Type::STRUCTREF:
                new (&type_struct_ref) decltype(type_struct_ref)(type.type_struct_ref);
                break;
            case TypeInfo::Type::STRUCT:
                new (&type_struct) decltype(type_struct)(type.type_struct);
                break;
            case TypeInfo::Type::FNREF:
                new (&type_fn_ref) decltype(type_fn_ref)(type.type_fn_ref);
                break;
            case TypeInfo::Type::FN:
                new (&type_fn) decltype(type_fn)(type.type_fn);
                break;
            case TypeInfo::Type::DEFREF:
                new (&type_def_ref) decltype(type_def_ref)(type.type_def_ref);
                break;
            case TypeInfo::Type::DEF:
                new (&type_def) decltype(type_def)(type.type_def);
                break;
            case TypeInfo::Type::INTRREF:
                new (&type_intr_ref) decltype(type_intr_ref)(type.type_intr_ref);
                break;
            case TypeInfo::Type::INTR:
                new (&type_intr) decltype(type_intr)(type.type_intr);
                break;
            case TypeInfo::Type::TENSOR:
            case TypeInfo::Type::EDGE:
                break;
            case TypeInfo::Type::GENERIC:
                new (&type_generic) decltype(type_generic)(type.type_generic);
                break;
            case TypeInfo::Type::ARRPACK:
                new (&type_arr_pack) decltype(type_arr_pack)(type.type_arr_pack);
                break;
            case TypeInfo::Type::AGGPACK:
                new (&type_agg_pack) decltype(type_agg_pack)(type.type_agg_pack);
                break;
            default:
                assert(false);
            }
        }

        TypeInfo TypeInfo::create_type(const TypeInfo& base, TypeInfo::Category cat)
        {
            TypeInfo type;
            new (&type.type_type) TypeInfoType();
            type.ty = TypeInfo::Type::TYPE;
            type.cat = cat;
            type.type_type.base = std::make_shared<TypeInfo>(base);
            return type;
        }

        TypeInfo TypeInfo::create_placeholder(TypeInfo::Category cat)
        {
            TypeInfo type;
            type.ty = TypeInfo::Type::PLACEHOLDER;
            type.cat = cat;
            return type;
        }

        TypeInfo TypeInfo::create_bool(TypeInfo::Category cat)
        {
            TypeInfo type;
            type.ty = TypeInfo::Type::BOOL;
            type.cat = cat;
            return type;
        }

        TypeInfo TypeInfo::create_int(TypeInfo::Category cat)
        {
            TypeInfo type;
            type.ty = TypeInfo::Type::INT;
            type.cat = cat;
            return type;
        }

        TypeInfo TypeInfo::create_float(TypeInfo::Category cat)
        {
            TypeInfo type;
            type.ty = TypeInfo::Type::FLOAT;
            type.cat = cat;
            return type;
        }

        TypeInfo TypeInfo::create_string(TypeInfo::Category cat)
        {
            TypeInfo type;
            type.ty = TypeInfo::Type::STR;
            type.cat = cat;
            return type;
        }

        TypeInfo TypeInfo::create_array(const TypeInfo& elem, TypeInfo::Category cat)
        {
            TypeInfo type;
            new (&type.type_array) TypeInfoArray();
            type.ty = TypeInfo::Type::ARRAY;
            type.cat = cat;
            type.type_array.elem = std::make_shared<TypeInfo>(elem);
            return type;
        }

        TypeInfo TypeInfo::create_tuple(const std::vector<TypeInfo>& elems, TypeInfo::Category cat)
        {
            TypeInfo type;
            new (&type.type_tuple) TypeInfoTuple();
            type.ty = TypeInfo::Type::TUPLE;
            type.cat = cat;
            type.type_tuple.elems = elems;
            return type;
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

        bool Scope::add(const std::string& var_name, const TypeInfo& info, const AstNodeInfo& node_info)
        {
            if (stack_vars.contains(var_name))
                return error::compiler(node_info, "");
            if (info.ty == TypeInfo::Type::TENSOR)
            stack_vars[var_name] = { info, 0 };
            return false;
        }

        void Scope::push(size_t n)
        {
            for (auto& [name, var] : stack_vars)
                var.ptr += n;
            if (parent)
                parent->push(n);
        }

        void Scope::pop(size_t n)
        {
            for (auto& [name, var] : stack_vars)
            {
                assert(var.ptr <= n);
                var.ptr -= n;
            }
            if (parent)
                parent->pop(n);
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

        std::string label_prefix(const AstNodeInfo& info)
        {
            char buf[64];
            sprintf(buf, "l%zuc%zu_", info.line_start, info.col_start);
            return buf;
        }

        CodeModule* pmod = nullptr;

        bool arg_type(TypeInfo& info, const Scope& scope, const AstArgDecl& arg)
        {
            TypeInfo explicit_type;
            if (arg.type_expr)
            {
                // TODO: determine the type from arg.type_expr
                return error::compiler(arg.node_info, "Internal error: not implemented");
            }

            TypeInfo default_type;
            if (arg.default_expr)
            {
                // TODO: determine the type of arg.default_expr
                return error::compiler(arg.node_info, "Internal error: not implemented");
            }

            if (explicit_type.ty == TypeInfo::Type::INVALID)
            {
                if (default_type.ty == TypeInfo::Type::INVALID)
                    return error::compiler(arg.node_info, "Missing both the type expression and the default expression");
                // only the default type was specified
                info = default_type;
                return false;
            }
            if (default_type.ty == TypeInfo::Type::INVALID)
            {
                // only the explicit type was specified
                info = explicit_type;
                return false;
            }
            // both the explicit and default type were specified
            if (explicit_type != default_type)
                return error::compiler(arg.node_info, "The default value's type for argument '%' did not match the explicitly declared type", arg.var_name);
            info = explicit_type;
            return false;
        }

        bool codegen_expr_bool(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            Obj obj;
            size_t addr;
            rets.push_back(TypeInfo::create_bool(TypeInfo::Category::CONST));
            return
                bc.heap.create_obj_bool(obj, expr.expr_bool) ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(expr.node_info, addr));
        }

        bool codegen_expr_int(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            Obj obj;
            size_t addr;
            rets.push_back(TypeInfo::create_int(TypeInfo::Category::CONST));
            return
                bc.heap.create_obj_int(obj, expr.expr_int) ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(expr.node_info, addr));
        }

        bool codegen_expr_float(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            Obj obj;
            size_t addr;
            rets.push_back(TypeInfo::create_float(TypeInfo::Category::CONST));
            return
                bc.heap.create_obj_float(obj, expr.expr_float) ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(expr.node_info, addr));
        }

        bool codegen_expr_string(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            Obj obj;
            size_t addr;
            rets.push_back(TypeInfo::create_string(TypeInfo::Category::CONST));
            return
                bc.heap.create_obj_str(obj, expr.expr_string) ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(expr.node_info, addr));
        }

        bool codegen_expr_array(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            if (expr.expr_agg.elems.size() == 0)
            {
                rets.push_back(TypeInfo::create_array(TypeInfo::create_placeholder()));
                return body.add_instruction(instruction::Agg(expr.node_info, 0));
            }
            size_t sz = 0;
            std::vector<TypeInfo> elem_types;
            for (const auto& elem_expr : expr.expr_agg.elems)
            {
                if (codegen_expr(bc, body, scope, elem_expr, elem_types))
                    return true;
                scope.push(elem_types.size() - sz);
                sz = elem_types.size();
            }
            return error::compiler(expr.node_info, "Internal error: not implemented");

            // TODO: Find a common element expression type between all of the elements and build ret
            return body.add_instruction(instruction::Agg(expr.node_info, sz));
        }

        bool codegen_expr_tuple(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            size_t sz = 0;
            std::vector<TypeInfo> elem_types;
            for (const auto& elem_expr : expr.expr_agg.elems)
            {
                if (codegen_expr(bc, body, scope, elem_expr, elem_types))
                    return true;
                scope.push(elem_types.size() - sz);
                sz = elem_types.size();
            }
            rets.push_back(TypeInfo::create_tuple(elem_types));
            return body.add_instruction(instruction::Agg(expr.node_info, sz));
        }

        bool codegen_expr_pos(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            std::vector<TypeInfo> results;
            if (codegen_expr(bc, body, scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The positive operator must be provided with a single numeric type, recieved '%' arguments", results.size());
            if (results[0].ty != TypeInfo::Type::INT && results[0].ty != TypeInfo::Type::FLOAT)
                return error::compiler(expr.node_info, "The positive operator must be provided with a single numeric type, recieved a '%' type", results[0].to_string());
            rets.push_back(results[0]);
            return false;  // the positive operator is a no-op
        }

        bool codegen_expr_neg(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            std::vector<TypeInfo> results;
            if (codegen_expr(bc, body, scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The negative operator must be provided with a single numeric type, recieved '%' arguments", results.size());
            if (results[0].ty != TypeInfo::Type::INT && results[0].ty != TypeInfo::Type::FLOAT)
                return error::compiler(expr.node_info, "The negative operator must be provided with a single numeric type, recieved a '%' type", results[0].to_string());
            // TODO: implement a neg instruction
            return error::compiler(expr.node_info, "Internal error: I forgot to implement a neg instruction");
        }

        bool codegen_expr_not(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            std::vector<TypeInfo> results;
            if (codegen_expr(bc, body, scope, *expr.expr_unary.expr, results))
                return true;
            if (results.size() != 1)
                return error::compiler(expr.node_info, "The not operator must be provided with a single boolean type, recieved '%' arguments", results.size());
            if (results[0].ty != TypeInfo::Type::BOOL)
                return error::compiler(expr.node_info, "The not operator must be provided with a single boolean type, recieved a '%' type", results[0].to_string());
            // TODO: implement a not instruction
            return error::compiler(expr.node_info, "Internal error: I forgot to implement a not instruction");
        }

        bool codegen_expr_unpack(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        constexpr char binop_errmsg[] = "The % hand side of % operation must be provided with a single value, recieved '%' values";

        bool codegen_expr_attempt_implicit(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, TypeInfo& base, TypeInfo& other)
        {
            constexpr char impcast_errmsg[] = "Type '%' cannot be implicitly cast to type '%'";

            switch (base.ty)
            {
            case TypeInfo::Type::INT:
                if (!other.check_xint())
                    return error::compiler(expr.node_info, impcast_errmsg, other.to_string(), "int");
                if (body.add_instruction(instruction::XInt(expr.node_info)))
                    return true;
                break;
            case TypeInfo::Type::FLOAT:
                if (!other.check_xflt())
                    return error::compiler(expr.node_info, impcast_errmsg, other.to_string(), "float");
                if (body.add_instruction(instruction::XFlt(expr.node_info)))
                    return true;
                break;
            case TypeInfo::Type::STR:
                if (!other.check_xstr())
                    return error::compiler(expr.node_info, impcast_errmsg, other.to_string(), "str");
                if (body.add_instruction(instruction::XStr(expr.node_info)))
                    return true;
                break;
            default:
                return error::compiler(expr.node_info, impcast_errmsg, other.to_string(), base.to_string());
            }
            return false;
        }
        
        template<class allowed>
        bool codegen_expr_single_ret(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            if (codegen_expr(bc, body, scope, expr, rets))
                return true;
            size_t prev_sz = rets.size();
            if (rets.size() - prev_sz != 1)
                return error::compiler(expr.node_info, "Expected a single value, recieved % values", rets.size() - prev_sz);
            if (!allowed::has_default && rets[0].cat == TypeInfo::Category::DEFAULT)
                return error::compiler(expr.node_info, "Expected a non-default value");
            if (!allowed::has_const && rets[0].cat == TypeInfo::Category::CONST)
                return error::compiler(expr.node_info, "Expected a non-constant value");
            if (!allowed::has_ref && rets[0].cat == TypeInfo::Category::REF)
                return error::compiler(expr.node_info, "Expected a non-reference value");
            if (!allowed::has_virtual && rets[0].cat == TypeInfo::Category::VIRTUAL)
                return error::compiler(expr.node_info, "Expected a non-virtual value");
            return false;
        }

        template<class lhs_allowed, class rhs_allowed, bool(*check)(TypeInfo&), const char* op_noun, const char* op_verb, class Op>
        bool codegen_expr_binop(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, TypeInfo& ret)
        {
            std::vector<TypeInfo> results;
            if (codegen_expr_single_ret<lhs_allowed>(bc, body, scope, *expr.expr_binary.left, results))
                return error::compiler(expr.node_info, "Unable to compile the left hand side of the % operation", op_noun);
            scope.push(1);
            if (codegen_expr_single_ret<rhs_allowed>(bc, body, scope, *expr.expr_binary.right, results))
                return error::compiler(expr.node_info, "Unable to compile the right hand side of the % operation", op_noun);
            scope.pop(1);
            if (!check(results[0]))
                return error::compiler(expr.node_info, "Unable to % values with type '%'", op_verb, results[0].to_string());

            if (results[0].ty != results[1].ty && codegen_expr_attempt_implicit(bc, body, scope, expr, results[0], results[1]))
                return true;
            ret = results[0];

            Obj obj;
            size_t addr;
            return
                results[0].to_obj(bc.heap, obj) ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(expr.node_info, addr)) ||
                body.add_instruction(Op(expr.node_info));
        }

        bool codegen_expr_add(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "addition";
            constexpr static char op_verb[] = "add";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_add(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Add>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back().cat = TypeInfo::Category::CONST;
            return false;
        }

        bool codegen_expr_sub(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "subtraction";
            constexpr static char op_verb[] = "subtract";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_sub(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Sub>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back().cat = TypeInfo::Category::CONST;
            return false;
        }

        bool codegen_expr_mul(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "multiplication";
            constexpr static char op_verb[] = "multiply";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_mul(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Mul>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back().cat = TypeInfo::Category::CONST;
            return false;
        }

        bool codegen_expr_div(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "division";
            constexpr static char op_verb[] = "divide";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_div(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Div>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back().cat = TypeInfo::Category::CONST;
            return false;
        }

        bool codegen_expr_mod(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "modulus";
            constexpr static char op_verb[] = "modulus";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_mod(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Mod>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back().cat = TypeInfo::Category::CONST;
            return false;
        }

        bool codegen_expr_iadd(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "addition assignment";
            constexpr static char op_verb[] = "add";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_add(); };
            return codegen_expr_binop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IAdd>(bc, body, scope, expr, rets.back());
        }

        bool codegen_expr_isub(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "subtraction assignment";
            constexpr static char op_verb[] = "subtract";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_sub(); };
            return codegen_expr_binop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::ISub>(bc, body, scope, expr, rets.back());
        }

        bool codegen_expr_imul(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "multiplication assignment";
            constexpr static char op_verb[] = "multiply";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_mul(); };
            return codegen_expr_binop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IMul>(bc, body, scope, expr, rets.back());
        }

        bool codegen_expr_idiv(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "division assignment";
            constexpr static char op_verb[] = "divide";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_div(); };
            return codegen_expr_binop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IDiv>(bc, body, scope, expr, rets.back());
        }

        bool codegen_expr_imod(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "modulus assignment";
            constexpr static char op_verb[] = "modulus";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_mod(); };
            return codegen_expr_binop<TypeInfo::Mutable, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::IMod>(bc, body, scope, expr, rets.back());
        }

        bool codegen_expr_assign(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_and(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_or(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_eq(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "equality";
            constexpr static char op_verb[] = "check the equality of";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_eq(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Eq>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_ne(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "inequality";
            constexpr static char op_verb[] = "check the inequality of";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_ne(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Ne>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_gt(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "greater than";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_gt(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Gt>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_lt(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "less than";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_lt(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Lt>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_ge(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "greater than or equal to";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_ge(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Ge>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_le(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            constexpr static char op_noun[] = "less than or equal to";
            constexpr static char op_verb[] = "compare";
            constexpr auto check_fn = [](TypeInfo& info) -> bool { return info.check_le(); };
            rets.push_back(TypeInfo());
            if (codegen_expr_binop<TypeInfo::NonVirtual, TypeInfo::NonVirtual, check_fn, op_noun, op_verb, instruction::Gt>(bc, body, scope, expr, rets.back()))
                return true;
            rets.back() = TypeInfo::create_bool(TypeInfo::Category::CONST);
            return false;
        }

        bool codegen_expr_idx(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_dot(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_decl(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_cargs(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_vargs(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_fndecl(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_defdecl(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_kw(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr_var(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            return error::compiler(expr.node_info, "Internal error: not implemented");
        }

        bool codegen_expr(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets)
        {
            switch (expr.ty)
            {
            case ExprType::INVALID:
                return error::compiler(expr.node_info, "Internal error: invalid AstExpr type");
            case ExprType::LIT_BOOL:
                return codegen_expr_bool(bc, body, scope, expr, rets);
            case ExprType::LIT_INT:
                return codegen_expr_int(bc, body, scope, expr, rets);
            case ExprType::LIT_FLOAT:
                return codegen_expr_float(bc, body, scope, expr, rets);
            case ExprType::LIT_STRING:
                return codegen_expr_string(bc, body, scope, expr, rets);
            case ExprType::LIT_ARRAY:
                return codegen_expr_array(bc, body, scope, expr, rets);
            case ExprType::LIT_TUPLE:
                return codegen_expr_tuple(bc, body, scope, expr, rets);
            case ExprType::UNARY_POS:
                return codegen_expr_pos(bc, body, scope, expr, rets);
            case ExprType::UNARY_NEG:
                return codegen_expr_neg(bc, body, scope, expr, rets);
            case ExprType::UNARY_NOT:
                return codegen_expr_not(bc, body, scope, expr, rets);
            case ExprType::UNARY_UNPACK:
                return codegen_expr_unpack(bc, body, scope, expr, rets);
            case ExprType::UNARY_REF:
                return error::compiler(expr.node_info, "Invalid usage of keyword 'ref'");
            case ExprType::UNARY_CONST:
                return error::compiler(expr.node_info, "Invalid usage of keyword 'const'");
            case ExprType::BINARY_ADD:
                return codegen_expr_add(bc, body, scope, expr, rets);
            case ExprType::BINARY_SUB:
                return codegen_expr_sub(bc, body, scope, expr, rets);
            case ExprType::BINARY_MUL:
                return codegen_expr_mul(bc, body, scope, expr, rets);
            case ExprType::BINARY_DIV:
                return codegen_expr_div(bc, body, scope, expr, rets);
            case ExprType::BINARY_MOD:
                return codegen_expr_mod(bc, body, scope, expr, rets);
            case ExprType::BINARY_IADD:
                return codegen_expr_iadd(bc, body, scope, expr, rets);
            case ExprType::BINARY_ISUB:
                return codegen_expr_isub(bc, body, scope, expr, rets);
            case ExprType::BINARY_IMUL:
                return codegen_expr_imul(bc, body, scope, expr, rets);
            case ExprType::BINARY_IDIV:
                return codegen_expr_idiv(bc, body, scope, expr, rets);
            case ExprType::BINARY_IMOD:
                return codegen_expr_imod(bc, body, scope, expr, rets);
            case ExprType::BINARY_ASSIGN:
                return codegen_expr_assign(bc, body, scope, expr, rets);
            case ExprType::BINARY_AND:
                return codegen_expr_and(bc, body, scope, expr, rets);
            case ExprType::BINARY_OR:
                return codegen_expr_or(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_EQ:
                return codegen_expr_eq(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_NE:
                return codegen_expr_ne(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_GT:
                return codegen_expr_gt(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_LT:
                return codegen_expr_lt(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_GE:
                return codegen_expr_ge(bc, body, scope, expr, rets);
            case ExprType::BINARY_CMP_LE:
                return codegen_expr_le(bc, body, scope, expr, rets);
            case ExprType::BINARY_IDX:
                return codegen_expr_idx(bc, body, scope, expr, rets);
            case ExprType::DOT:
                return codegen_expr_dot(bc, body, scope, expr, rets);
            case ExprType::VAR_DECL:
                return codegen_expr_decl(bc, body, scope, expr, rets);
            case ExprType::CARGS_CALL:
                return codegen_expr_cargs(bc, body, scope, expr, rets);
            case ExprType::VARGS_CALL:
                return codegen_expr_vargs(bc, body, scope, expr, rets);
            case ExprType::FN_DECL:
                return codegen_expr_fndecl(bc, body, scope, expr, rets);
            case ExprType::DEF_DECL:
                return codegen_expr_defdecl(bc, body, scope, expr, rets);
            case ExprType::KW:
                return codegen_expr_kw(bc, body, scope, expr, rets);
            case ExprType::VAR:
                return codegen_expr_var(bc, body, scope, expr, rets);
            }
            return error::compiler(expr.node_info, "Internal error: unknown AstExpr type");
        }

        bool codegen_exit(ByteCodeBody& body, Scope& scope, const AstNodeInfo& info)
        {
            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, scope.parent))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body.add_instruction(instruction::Pop(info, var.ptr)))
                    return true;
            }
            scope.parent->pop(vars.size());
            return false;
        }

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

        bool codegen_line_break(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "break statements are not allowed outside a looping structure");

            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, loop_ctx->scope))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body.add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            }
            return body.add_instruction(instruction::Jmp(line.node_info, loop_ctx->break_label));
        }

        bool codegen_line_continue(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            if (!loop_ctx)
                return error::compiler(line.node_info, "continue statements are not allowed outside a looping structure");

            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, loop_ctx->scope))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
            {
                if (body.add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            }
            return body.add_instruction(instruction::Jmp(line.node_info, loop_ctx->break_label));
        }

        bool codegen_line_export(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_extern(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_raise(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            std::vector<TypeInfo> rets;
            if (codegen_expr(bc, body, scope, line.line_func.expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(line.node_info, "A raise expression must resolve to a single string value");
            if (rets[0].ty != TypeInfo::Type::STR)
                return body.add_instruction(instruction::Err(line.node_info));
            else if (!rets[0].check_xstr())
                return error::compiler(line.node_info, "A raise expression must resolve to a single string value");
            return
                body.add_instruction(instruction::XStr(line.node_info)) ||
                body.add_instruction(instruction::Err(line.node_info));
        }

        bool codegen_line_print(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            std::vector<TypeInfo> rets;
            if (codegen_expr(bc, body, scope, line.line_func.expr, rets))
                return true;
            if (rets.size() != 1)
                return error::compiler(line.node_info, "A print expression must resolve to a single string value");
            if (rets[0].ty != TypeInfo::Type::STR)
                return body.add_instruction(instruction::Dsp(line.node_info));
            else if (!rets[0].check_xstr())
                return error::compiler(line.node_info, "A print expression must resolve to a single string value");
            return
                body.add_instruction(instruction::XStr(line.node_info)) ||
                body.add_instruction(instruction::Dsp(line.node_info));
        }

        bool codegen_line_return(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            // TODO: codegen the return value using the signature to disambiguate tuple return values vs multiple return values
            std::vector<Scope::StackVar> vars;
            if (scope.list_local_vars(vars, nullptr))
                return true;
            std::sort(vars.begin(), vars.end(), [](const Scope::StackVar& lhs, const Scope::StackVar& rhs) { return lhs.ptr > rhs.ptr; });
            for (const auto& var : vars)
                if (body.add_instruction(instruction::Pop(line.node_info, var.ptr)))
                    return true;
            if (ret_label)
                return body.add_instruction(instruction::Jmp(line.node_info, *ret_label));
            return body.add_instruction(instruction::Ret(line.node_info));
        }

        bool codegen_line_branch(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line, const std::string& end_label)
        {
            Scope block_scope{ &scope };
            Scope cond_scope{ &scope };
            std::vector<TypeInfo> rets;
            if (codegen_expr(bc, body, cond_scope, line.line_branch.cond, rets) ||
                codegen_exit(body, cond_scope, line.node_info)  // leaving the conditional's scope
                ) return true;
            if (rets.size() != 1 || rets[0].ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a single boolean value");
            std::string false_branch = label_prefix(line.node_info) + "false_branch";
            return
                body.add_instruction(instruction::Brf(line.node_info, false_branch)) ||
                codegen_lines(bc, body, block_scope, line.line_branch.body) ||
                codegen_exit(body, block_scope, line.node_info) || // leaving the block's scope
                body.add_instruction(instruction::Jmp(line.node_info, end_label)) ||
                body.add_label(line.node_info, false_branch);
        }

        bool codegen_line_while(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            LoopContext* old_ctx = loop_ctx;
            Scope cond_scope{ &scope };
            Scope block_scope{ &scope };
            std::string loop_start = label_prefix(line.node_info) + "loop_start";
            std::string loop_end = label_prefix(line.node_info) + "loop_end";
            std::vector<TypeInfo> rets;
            if (body.add_label(line.node_info, loop_start) ||
                codegen_expr(bc, body, cond_scope, line.line_branch.cond, rets) ||
                codegen_exit(body, cond_scope, line.node_info)
                ) return true;
            LoopContext new_ctx = { &block_scope, loop_start, loop_end };
            loop_ctx = &new_ctx;
            if (rets.size() != 1 || rets[0].ty != TypeInfo::Type::BOOL)
                return error::compiler(line.node_info, "A conditional expression must resolve to a single boolean value");
            bool ret =
                body.add_instruction(instruction::Brf(line.node_info, loop_end)) ||
                codegen_lines(bc, body, block_scope, line.line_branch.body) ||
                body.add_instruction(instruction::Jmp(line.node_info, loop_start)) ||
                body.add_label(line.node_info, loop_end);
            loop_ctx = old_ctx;
            return ret;
        }

        bool codegen_line_for(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            return error::compiler(line.node_info, "Internal error: not implemented");
        }

        bool codegen_line_expr(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            std::vector<TypeInfo> rets;
            if (codegen_expr(bc, body, scope, line.line_expr.line, rets))
                return true;
            // getting rid of the immediate rets, not local variables
            for (const TypeInfo& info : rets)
            {
                if (body.add_instruction(instruction::Pop(line.node_info, 0)))
                    return true;
            }
            return false;
        }

        bool codegen_line(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line)
        {
            switch (line.ty)
            {
            case LineType::INVALID:
                return error::compiler(line.node_info, "Internal error: invalid AstLine type");
            case LineType::BREAK:
                return codegen_line_break(bc, body, scope, line);
            case LineType::CONTINUE:
                return codegen_line_continue(bc, body, scope, line);
            case LineType::EXPORT:
                return codegen_line_export(bc, body, scope, line);
            case LineType::EXTERN:
                return codegen_line_extern(bc, body, scope, line);
            case LineType::RAISE:
                return codegen_line_raise(bc, body, scope, line);
            case LineType::PRINT:
                return codegen_line_print(bc, body, scope, line);
            case LineType::RETURN:
                return codegen_line_return(bc, body, scope, line);
            case LineType::IF:
                return error::compiler(line.node_info, "Internal error: recieved dependent line type 'if' in codegen_line");
            case LineType::ELIF:
                return error::compiler(line.node_info, "Found elif statement without a matching if statement");
            case LineType::ELSE:
                return error::compiler(line.node_info, "Found else statement without a matching if statement");
            case LineType::WHILE:
                return codegen_line_while(bc, body, scope, line);
            case LineType::FOR:
                return codegen_line_for(bc, body, scope, line);
            case LineType::EXPR:
                return codegen_line_expr(bc, body, scope, line);
            case LineType::FORWARD:
                return error::compiler(line.node_info, "Internal error: recieved dependent line type 'forward' in codegen_line");
            case LineType::BACKWARD:
                return error::compiler(line.node_info, "Found backward statement without a matching forward statement");
            case LineType::EVALMODE:
                return error::compiler(line.node_info, "Internal error: not implemented");
            default:
                return error::compiler(line.node_info, "Internal error: unknown AstLine type");
            }
        }

        bool codegen_lines(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const std::vector<AstLine>& lines)
        {
            assert(lines.size() > 0);

            size_t i = 0;
            while (i < lines.size())
            {
                if (lines[i].ty == LineType::IF)
                {
                    std::string end_label = label_prefix(lines[i].node_info) + "branch_end";
                    if (codegen_line_branch(bc, body, scope, lines[i], end_label))
                        return true;
                    while (i < lines.size() && lines[i].ty == LineType::ELIF)
                    {
                        if (codegen_line_branch(bc, body, scope, lines[i], end_label))
                            return true;
                        i++;
                    }
                    if (i < lines.size() && lines[i].ty == LineType::ELSE)
                    {
                        if (codegen_lines(bc, body, scope, lines[i].line_block.body))
                            return true;
                    }
                    body.add_label(lines[i].node_info, end_label);
                    i++;
                }
                else if (lines[i].ty == LineType::FORWARD)
                {
                    return error::compiler(lines[i].node_info, "Internal error: not implemented");
                }
                else
                    codegen_line(bc, body, scope, lines[i]);
            }

            // During proper code execution, this point should be unreachable.
            // If this point is reached during execution, it will fall through to the next function which is undefined
            // and will cause the interpreter to start bahaving unpredictably which would be very difficult to debug.
            // For safety, a raise statement is added manually by the compiler to generate a runtime error
            // if the programmer made a mistake.  If this point was in fact unreachable,
            // the raise will never get run and will not have any consequences (besides moving labels around in the bytecode)
            Obj obj;
            size_t addr;
            return
                bc.heap.create_obj_str(obj, "Reached the end of the procedure without returning") ||
                bc.add_static_obj(obj, addr) ||
                body.add_instruction(instruction::New(lines[lines.size() - 1].node_info, addr)) ||
                body.add_instruction(instruction::Err(lines[lines.size() - 1].node_info));
        }

        bool codegen_struct(ByteCodeModule& bc, const std::string& name, const AstStruct& ast_struct, const std::vector<std::string>& ns)
        {
            ByteCodeBody body{ ast_struct.node_info };

        }

        bool codegen_func(ByteCodeModule& bc, const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns)
        {
            ByteCodeBody body{ ast_fn.node_info };
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
                TypeInfo info;
                if (arg_type(info, scope, arg) ||
                    scope.add(arg.var_name, info, arg.node_info)
                    ) return true;
                fn_name << info.encode() << "_";
            }
            for (const auto& arg : ast_fn.signature.vargs)
            {
                TypeInfo info;
                if (arg_type(info, scope, arg) ||
                    scope.add(arg.var_name, info, arg.node_info)
                    ) return true;
                fn_name << info.encode() << "_";
            }
            if (scope.add("~block", TypeInfo::create_block(), ast_fn.node_info))
                return true;
            fn_name << name;
            return
                codegen_lines(bc, body, scope, ast_fn.body) ||
                bc.add_block(fn_name.str(), body);
        }

        bool codegen_def(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns)
        {
            ByteCodeBody body{ ast_def.node_info };
            Scope scope{ nullptr };
            std::stringstream def_name;
            def_name << "def_";
            for (const auto& ns_name : ns)
                def_name << ns_name << "_";
            def_name
                << ast_def.signature.cargs.size() << "_"
                << ast_def.signature.vargs.size() << "_";

            for (const auto& arg : ast_def.signature.cargs)
            {
                TypeInfo info;
                if (arg_type(info, scope, arg) ||
                    scope.add(arg.var_name, info, arg.node_info)
                    ) return true;
                def_name << info.encode() << "_";
            }
            for (const auto& arg : ast_def.signature.vargs)
            {
                TypeInfo info;
                if (arg_type(info, scope, arg))
                    return true;
                if (info.ty != TypeInfo::Type::TENSOR)
                    return error::compiler(arg.node_info, "def arguments must be tensors");
                if (scope.add(arg.var_name, info, arg.node_info))
                    return true;
            }
            if (scope.add("~block", TypeInfo::create_block(), ast_def.node_info))
                return true;
            def_name << name;

            // Creating the new block
            Obj def_name_obj;
            size_t def_name_addr;
            if (bc.heap.create_obj_str(def_name_obj, name) ||
                bc.add_static_obj(def_name_obj, def_name_addr) ||
                body.add_instruction(instruction::New(ast_def.node_info, def_name_addr)) ||
                body.add_instruction(instruction::Blk(ast_def.node_info)) ||
                body.add_instruction(instruction::BkPrt(ast_def.node_info))
                ) return true;

            // Adding the block configurations
            for (const auto& arg : ast_def.signature.cargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve def carg '%'", arg.var_name);
                if (var.info.runtime_obj())
                {
                    Obj obj;
                    size_t addr;
                    if (body.add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                        var.info.to_obj(bc.heap, obj) ||
                        bc.add_static_obj(obj, addr) ||
                        body.add_instruction(instruction::New(arg.node_info, addr)) ||
                        bc.heap.create_obj_str(obj, arg.var_name) ||
                        bc.add_static_obj(obj, addr) ||
                        body.add_instruction(instruction::New(arg.node_info, addr)) ||
                        body.add_instruction(instruction::BkCfg(arg.node_info))
                        ) return true;
                }
            }

            // Adding block inputs
            for (const auto& arg : ast_def.signature.vargs)
            {
                Scope::StackVar var;
                if (scope.at(arg.var_name, var))
                    return error::compiler(arg.node_info, "Internal error: unable to retrieve def varg '%'", arg.var_name);
                assert(var.info.ty == TypeInfo::Type::TENSOR);
                Obj obj;
                size_t addr;
                if (body.add_instruction(instruction::Dup(arg.node_info, var.ptr)) ||
                    bc.heap.create_obj_str(obj, arg.var_name) ||
                    bc.add_static_obj(obj, addr) ||
                    body.add_instruction(instruction::New(arg.node_info, addr)) ||
                    body.add_instruction(instruction::BkInp(arg.node_info))
                    ) return true;
            }

            std::string def_ret_label = label_prefix(ast_def.node_info) + "_return";
            ret_label = &def_ret_label;
            if (codegen_lines(bc, body, scope, ast_def.body))
                return true;
            ret_label = nullptr;

            if (body.add_label(ast_def.node_info, def_ret_label) ||
                body.add_instruction(instruction::Dup(ast_def.node_info, ast_def.signature.rets.size()))  // grabbing a reference to the block
                ) return true;

            // Adding block outputs
            for (size_t i = 0; i < ast_def.signature.rets.size(); i++)
            {
                Obj obj;
                size_t addr;
                if (body.add_instruction(instruction::Dup(ast_def.node_info, ast_def.signature.rets.size() - i)) ||  // no -1 to account for the block reference
                    bc.heap.create_obj_str(obj, ast_def.signature.rets[i]) ||
                    bc.add_static_obj(obj, addr) ||
                    body.add_instruction(instruction::New(ast_def.node_info, addr)) ||
                    body.add_instruction(instruction::BkOut(ast_def.node_info))
                    ) return true;
            }

            return
                body.add_instruction(instruction::Pop(ast_def.node_info, 0)) ||  // popping the block off for the ret
                body.add_instruction(instruction::Ret(ast_def.node_info)) ||
                bc.add_block(def_name.str(), body);
        }

        bool codegen_intr(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_intr, const std::vector<std::string&> ns)
        {

        }

        bool codegen_attr(ByteCodeModule& bc, const std::string& name, const CodeModule::Attr& attr, std::vector<std::string>& ns)
        {
            switch (attr.index())
            {
            case CodeModule::AttrType::NODE:
                for (const auto& [node_name, node_attrs] : std::get<CodeModule::Node>(attr).attrs)
                {
                    ns.push_back(node_name);
                    for (const auto& node_attr : node_attrs)
                        if (codegen_attr(bc, node_name, node_attr, ns))
                            return true;
                    ns.pop_back();
                }
            case CodeModule::AttrType::STRUCT:
                return codegen_struct(bc, name, std::get<AstStruct>(attr), ns);
            case CodeModule::AttrType::FUNC:
                return codegen_func(bc, name, std::get<AstFn>(attr), ns);
            case CodeModule::AttrType::DEF:
                return codegen_def(bc, name, std::get<AstBlock>(attr), ns);
            }
            assert(false);
            return true;
        }

        bool codegen_module(ByteCodeModule& bc, const AstModule& ast, const std::vector<std::string>& imp_dirs)
        {
            // Resolving imports to build a CodeModule object
            CodeModule mod;
            std::vector<std::string> visited = { ast.fname };
            if (CodeModule::create(mod, ast, imp_dirs, visited))
                return true;
            pmod = &mod;

            std::vector<std::string> ns;
            for (const auto& [name, attrs] : mod.root.attrs)
                for (const auto& attr : attrs)
                    if (codegen_attr(bc, name, attr, ns))
                        return true;
            return false;
        }
    }
}
