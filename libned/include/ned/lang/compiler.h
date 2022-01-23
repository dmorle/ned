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
            std::vector<AstStruct> matches;
        };

        struct TypeInfoStruct
        {
            std::vector<AstStruct> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoFnRef
        {
            std::vector<AstFn> matches;
        };

        struct TypeInfoFn
        {
            std::vector<AstFn> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoDefRef
        {
            std::vector<AstBlock> matches;
        };

        struct TypeInfoDef
        {
            std::vector<AstBlock> matches;
            std::vector<TypeInfo> cargs;
            std::map<std::string, TypeInfo> kwcargs;
        };

        struct TypeInfoIntrRef
        {
            std::vector<AstBlock> matches;
        };

        struct TypeInfoIntr
        {
            std::vector<AstBlock> matches;
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
            enum
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
                TENSOR,      // Both a forward and backward edge adjacent in the stack
                EDGE,        // Either forward or backward edge
                GENERIC,     // Compile time generic type
                ARRPACK,     // A packed set of parameters (array of a single type)
                AGGPACK      // The return value of a non-struct callable (tuple of types)
            } ty = TypeInfo::INVALID;

            bool is_const;
            bool is_ref;

            bool xstr();  // whether the type can be converted into a string
            bool xint();  // whether the type can be converted into an int
            bool xflt();  // whether the type can be converted into a float

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

            TypeInfo(const TypeInfo& type);
            TypeInfo& operator=(const TypeInfo& type);
            ~TypeInfo();
        
        private:
            TypeInfo();

        public:
            static TypeInfo create_type(const TypeInfo& base);
            static TypeInfo create_placeholder();
            static TypeInfo create_bool();
            static TypeInfo create_int();
            static TypeInfo create_float();
            static TypeInfo create_string();
            static TypeInfo create_array(const TypeInfo& elem);
            static TypeInfo create_tuple(const std::vector<TypeInfo>& elems);
        };

        class Scope
        {
            friend bool codegen_exit(ByteCodeBody& body, Scope& scope, const AstNodeInfo& info);

        public:
            struct StackVar
            {
                TypeInfo info;
                size_t ptr;  // For tensor types, the backward edge will always be at ptr-1
            };

        private:
            std::map<std::string, StackVar> stack_vars;
            Scope* parent = nullptr;

        public:
            bool at(const std::string& var_name, StackVar& var) const;
            bool add(const std::string& var_name, const TypeInfo& info);

            bool push(size_t n);
            bool pop(size_t n);

            // This function does not return the actual size on the stack thats used by the scope
            // Instead, they provide the number of types that are stored on the stack,
            // To determine the total size on the stack, accumulate the size for each type
            bool local_size(size_t& sz, const Scope* scope) const;
            bool list_local_vars(std::vector<StackVar>& vars, const Scope* scope);

            static bool create_sub(Scope& scope, Scope* parent);
            static bool create_struct(Scope& scope, AstStructSig& sig);
            static bool create_func(Scope& scope, AstFnSig& sig);
            static bool create_def(Scope& scope, AstBlockSig& sig);
            static bool create_intr(Scope& scope, AstBlockSig& sig);
        };

        std::string label_prefix(const AstNodeInfo& info);

        bool expr_type(TypeInfo& info, const CodeModule& mod, const Scope* scope, const AstExpr& ast);

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
        bool codegen_loop(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const std::vector<AstLine>& lines, const std::string& cont_lbl, const std::string& break_lbl);
        bool codegen_lines(ByteCodeModule& bc, ByteCodeBody& body, Scope& scope, const std::vector<AstLine>& lines, bool fallthrough);

        bool codegen_struct(ByteCodeModule& bc, const std::string& name, const AstStruct& ast_struct, const std::vector<std::string>& ns);
        bool codegen_func(ByteCodeModule& bc, const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns);
        bool codegen_def(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns);
        bool codegen_intr(ByteCodeModule& bc, const std::string& name, const AstBlock& ast_intr, const std::vector<std::string&> ns);
        bool codegen_attr(ByteCodeModule& bc, const std::string& name, const CodeModule::Attr& attr, std::vector<std::string>& ns);

        bool codegen_module(ByteCodeModule& bc, const AstModule& ast, const std::vector<std::string>& imp_dirs);
    }
}

#endif
