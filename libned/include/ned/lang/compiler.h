#ifndef NED_TYPING_H
#define NED_TYPING_H

#include <ned/lang/ast.h>
#include <ned/lang/bytecode.h>

#include <variant>

namespace nn
{
    namespace lang
    {
        // Resolves imports of an AstModule to build a single code module without external dependancies
        class CodeModule
        {
        public:
            struct Node;
            using Attr = std::variant<Node, AstStruct, AstFn, AstBlock>;
            enum AttrType
            {
                NODE = 0,
                STRUCT,
                FUNC,
                DEF
            };

            struct Node
            {
                std::map<std::string, std::vector<Attr>>     attrs;
                std::map<std::string, std::vector<AstBlock>> intrs;
                std::map<std::string, std::vector<AstInit>>  inits;
            };

        private:
            template<typename T>
            static bool merge_node(Node& dst, T& src);

        public:
            Node root;
            bool merge_ast(AstModule& ast);
            static bool create(CodeModule& mod, const AstModule& ast, const std::vector<std::string>& imp_dirs, std::vector<std::string> visited);
        };

        struct TypeInfo;

        struct TypeInfoType
        {
            std::shared_ptr<TypeInfo> base;
        };

        struct TypeInfoArray
        {
            std::shared_ptr<TypeInfo> elem;
        };

        struct TypeInfoTuple
        {
            std::vector<TypeInfo> elems;
        };

        struct TypeInfoNamespace
        {
            std::shared_ptr<CodeModule::Node> node;
        };

        struct TypeInfoStructRef
        {
            std::vector<const AstStruct*> matches;
        };

        struct TypeInfoStruct
        {
            std::vector<const AstStruct*> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoFnRef
        {
            std::vector<const AstFn*> matches;
        };

        struct TypeInfoFn
        {
            std::vector<const AstFn*> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoDefRef
        {
            std::vector<const AstBlock*> matches;
        };

        struct TypeInfoDef
        {
            std::vector<const AstBlock*> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoIntrRef
        {
            std::vector<const AstBlock*> matches;
        };

        struct TypeInfoIntr
        {
            std::vector<const AstBlock*> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoGeneric
        {
            std::string name;
        };

        struct TypeInfoArrPack
        {
            std::shared_ptr<TypeInfo> elem;
        };

        struct TypeInfoAggPack
        {
            std::vector<TypeInfo> elems;
        };

        struct TypeInfo
        {
            enum class Type
            {
                INVALID = 0, // Invalid (uninitialized) type object
                TYPE,        // Type object
                PLACEHOLDER, // Used as the element type for the array literal []
                BOOL,        // Boolean
                FWIDTH,      // Float widths for tensors, ie f16, f32, f64
                INT,         // Integer
                FLOAT,       // Floating point
                STR,         // String
                ARRAY,       // Array - Fixed length
                TUPLE,       // Tuple - mainly for cargs
                NAMESPACE,   // Reference to a namespace
                STRUCTREF,   // Reference to a struct name
                STRUCT,      // Struct with bound cargs
                FNREF,       // Reference to a function name
                FN,          // Function with bound cargs
                DEFREF,      // Reference to a block name
                DEF,         // Block with bound cargs
                INTRREF,     // Reference to a intrinsic name
                INTR,        // Intrinsic with bound cargs
                INITREF,     // Reference to an init name
                INIT,        // Init object
                NODE,        // Node object
                EDGE,        // Edge object
                BLOCK,       // Block object
                TENSOR,      // Both a forward and backward edge
                GENERIC,     // Compile time generic type
                ARRPACK,     // A packed set of parameters (array of a single type)
                AGGPACK      // The return value of a non-struct callable (tuple of types)
            } ty = TypeInfo::Type::INVALID;

            enum class Category
            {
                DEFAULT,
                CONST,
                REF,
                VIRTUAL
            } cat = TypeInfo::Category::DEFAULT;

            template<Category...>
            struct CategorySet;

            template<Category head, Category... tail>
            struct CategorySet<head, tail...>
            {
                constexpr static bool has_default = head == Category::DEFAULT || CategorySet<tail...>::has_default;
                constexpr static bool has_const   = head == Category::CONST   || CategorySet<tail...>::has_const;
                constexpr static bool has_ref     = head == Category::REF     || CategorySet<tail...>::has_ref;
                constexpr static bool has_virtual = head == Category::VIRTUAL || CategorySet<tail...>::has_virtual;
            };

            template<>
            struct CategorySet<>
            {
                constexpr static bool has_default = false;
                constexpr static bool has_const   = false;
                constexpr static bool has_ref     = false;
                constexpr static bool has_virtual = false;
            };

            using AllowAll = CategorySet<Category::DEFAULT, Category::CONST, Category::REF, Category::VIRTUAL>;
            using NonVirtual = CategorySet<Category::DEFAULT, Category::CONST, Category::REF>;
            using NonConst = CategorySet<Category::DEFAULT, Category::REF, Category::VIRTUAL>;
            using Mutable = CategorySet<Category::DEFAULT, Category::REF>;

            bool check_add();
            bool check_sub();
            bool check_mul();
            bool check_div();
            bool check_mod();

            bool check_eq();
            bool check_ne();
            bool check_ge();
            bool check_le();
            bool check_gt();
            bool check_lt();

            bool check_xstr();  // whether the type can be converted into a string
            bool check_xint();  // whether the type can be converted into an int
            bool check_xflt();  // whether the type can be converted into a float

            union
            {
                TypeInfoType      type_type;
                TypeInfoArray     type_array;
                TypeInfoTuple     type_tuple;
                TypeInfoNamespace type_namespace;
                TypeInfoStructRef type_struct_ref;
                TypeInfoStruct    type_struct;
                TypeInfoFnRef     type_fn_ref;
                TypeInfoFn        type_fn;
                TypeInfoDefRef    type_def_ref;
                TypeInfoDef       type_def;
                TypeInfoIntrRef   type_intr_ref;
                TypeInfoIntr      type_intr;
                TypeInfoGeneric   type_generic;
                TypeInfoArrPack   type_arr_pack;
                TypeInfoAggPack   type_agg_pack;
            };

            TypeInfo();
            TypeInfo(const TypeInfo& type);
            TypeInfo& operator=(const TypeInfo& type);
            ~TypeInfo();
        
            static TypeInfo create_type(const TypeInfo& base, TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_placeholder(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_bool(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_int(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_float(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_string(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_array(const TypeInfo& elem, TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_tuple(const std::vector<TypeInfo>& elems, TypeInfo::Category cat = TypeInfo::Category::DEFAULT);

            static TypeInfo create_init(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_node(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_edge(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_block(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);
            static TypeInfo create_tensor(TypeInfo::Category cat = TypeInfo::Category::DEFAULT);

            std::string encode();
            void decode(const std::string& data);
            std::string to_string();
            bool runtime_obj();  // returns whether or not the type can be converted into a runtime object
            bool to_obj(ProgramHeap& heap, Obj& obj);  // converts the type into a runtime object

        private:
            void do_copy(const TypeInfo& type);
        };

        bool operator==(const TypeInfo& lhs, const TypeInfo& rhs);

        class Scope
        {
            friend bool codegen_exit(ByteCodeBody& body, Scope& scope, const AstNodeInfo& info);

        public:
            struct StackVar
            {
                TypeInfo info;
                size_t ptr;
            };

        private:
            std::map<std::string, StackVar> stack_vars;
            Scope* parent = nullptr;

        public:
            Scope(Scope* parent) : parent(parent) {}

            bool at(const std::string& var_name, StackVar& var) const;
            bool add(const std::string& var_name, const TypeInfo& info, const AstNodeInfo& node_info);

            void push(size_t n);
            void pop(size_t n);

            bool local_size(size_t& sz, const Scope* scope) const;
            bool list_local_vars(std::vector<StackVar>& vars, const Scope* scope);
        };

        std::string label_prefix(const AstNodeInfo& info);

        bool arg_type(TypeInfo& info, const Scope& scope, const AstArgDecl& arg);

        bool codegen_expr_bool     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_int      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_float    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_string   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_array    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_tuple    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_pos      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_neg      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_not      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_unpack   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_add      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_sub      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_mul      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_div      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_mod      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_iadd     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_isub     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_imul     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_idiv     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_imod     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_assign   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_and      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_or       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_eq       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_ne       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_gt       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_lt       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_ge       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_le       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_idx      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_dot      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_decl     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_cargs    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_vargs    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_fndecl   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_defdecl  (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_kw       (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_expr_var      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);

        bool codegen_expr(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstExpr& expr, std::vector<TypeInfo>& rets);
        bool codegen_exit(ByteCodeBody& body, Scope& scope, const AstNodeInfo& info);  // unwinds and exits a scope updating the parent scope's pointers

        bool codegen_line_break    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_continue (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_export   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_extern   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_raise    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_print    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_return   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_branch   (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line, const std::string& end_label);  // if or elif statement
        bool codegen_line_block    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_while    (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_for      (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);
        bool codegen_line_expr     (ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);

        bool codegen_line(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const AstLine& line);  // generates code for independent lines.  Can't handle if, elif, else, forward.
        bool codegen_lines(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const std::vector<AstLine>& lines);

        bool codegen_struct(ByteCodeModule& bc, const std::string& name, const AstStruct& ast_struct, const std::vector<std::string>& ns);
        bool codegen_func(ByteCodeModule& bc, const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns);
        bool codegen_def(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns);
        bool codegen_intr(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_intr, const std::vector<std::string&> ns);
        bool codegen_attr(ByteCodeModule& bc, const std::string& name, const CodeModule::Attr& attr, std::vector<std::string>& ns);

        bool codegen_module(ByteCodeModule& bc, const AstModule& ast, const std::vector<std::string>& imp_dirs);
    }
}

#endif
