#ifndef NED_TYPING_H
#define NED_TYPING_H

#include <ned/lang/ast.h>
#include <ned/lang/bytecode.h>

#include <variant>
#include <functional>

namespace nn
{
    namespace lang
    {
        // Resolves imports of an AstModule to build a single code module without external dependancies
        class CodeModule
        {
        public:
            struct Namespace
            {
                std::map<std::string, std::vector<Namespace>>        namespaces;
                std::map<std::string, std::vector<const AstStruct*>> structs;
                std::map<std::string, std::vector<const AstEnum*>>   enums;
                std::map<std::string, std::vector<const AstFn*>>     fns;
                std::map<std::string, std::vector<const AstBlock*>>  defs;
                std::map<std::string, std::vector<const AstBlock*>>  intrs;
                std::map<std::string, std::vector<const AstInit*>>   inits;
            };

            struct LookupResult
            {
                std::vector<std::string> ns;  // the signature's namespace
                std::vector<const Namespace*> namespaces;
                std::vector<const AstStruct*> structs;
                std::vector<const AstEnum*>   enums;
                std::vector<const AstFn*>     fns;
                std::vector<const AstBlock*>  defs;
                std::vector<const AstBlock*>  intrs;
                std::vector<const AstInit*>   inits;

                bool empty();
            };

        private:
            template<typename T>
            static bool merge_namespace(Namespace& dst, T& src);

        public:
            Namespace root;
            bool merge_ast(AstModule& ast);
            static bool create(CodeModule& mod, AstModule& ast, const std::vector<std::string>& imp_dirs, std::vector<std::string> visited);

            struct LookupCtx
            {
                Namespace& nd;
                std::vector<std::string>::const_iterator it;
                std::vector<std::string>::const_iterator end;
            };

            static bool lookup(const LookupCtx& ctx, const std::string& idn, LookupResult& result);
        };

        class TypeInfo;
        class Scope;

        class TypeRef
        {
            friend class TypeManager;
            friend class TypeInfo;

            size_t ptr;
            TypeRef(size_t ptr);

        public:
            TypeRef() : ptr(0) {}
            TypeRef(const TypeRef&) = default;
            TypeRef(TypeRef&&) = default;
            TypeRef& operator=(const TypeRef&) = default;
            TypeRef& operator=(TypeRef&&) = default;

            operator bool() const noexcept;
            TypeInfo* operator->() noexcept;
            const TypeInfo* operator->() const noexcept;
            TypeInfo& operator*() noexcept;
            const TypeInfo& operator*() const noexcept;

        };

        using CodegenCallback = std::function<bool(Scope&)>;

        struct TypeInfoType
        {
            TypeRef base;
        };

        struct TypeInfoArray
        {
            TypeRef elem;
        };

        struct TypeInfoTuple
        {
            std::vector<TypeRef> elems;
        };

        struct TypeInfoLookup
        {
            std::string name;
            CodeModule::LookupResult lookup;
        };

        struct TypeInfoCargBind
        {
            std::string name;
            CodeModule::LookupResult lookup;
            const std::vector<AstExpr>& cargs;
        };

        struct TypeInfoStruct
        {
            const AstStruct* ast;
            std::map<std::string, TypeRef> cargs;
            std::vector<std::pair<std::string, TypeRef>> fields;
            bool lazy;
        };

        struct TypeInfoEnum
        {
            const AstEnum* ast;
            std::vector<std::string> ast_ns;
            std::map<std::string, TypeRef> cargs;
        };

        struct TypeInfoEnumEntry  // Result of a non-trivial enum entry access (always virtual)
        {
            const AstEnum* ast;
            std::vector<std::string> ast_ns;
            std::map<std::string, TypeRef> cargs;
            size_t idx;
        };

        struct TypeInfoDlType
        {
            TypeRef tensor;
            TypeRef edge;
            TypeRef shape;
            TypeRef fp;
        };

        struct TypeInfoGeneric
        {
            std::string name;
        };

        struct TypeInfoArrPack
        {
            TypeRef elem;
        };

        struct TypeInfoAggPack
        {
            std::vector<TypeRef> elems;
        };

        class TypeInfo
        {
        public:
            const static TypeRef null;

            enum class Type
            {
                INVALID = 0, // Invalid (uninitialized) type object
                TYPE,        // Type object
                PLACEHOLDER, // Used as the element type for the array literal []
                FTY,         // Float widths for tensors, ie f16, f32, f64
                BOOL,        // Boolean
                INT,         // Integer
                FLOAT,       // Floating point
                STR,         // String
                CFG,         // Config

                ARRAY,       // Array
                TUPLE,       // Tuple
                LOOKUP,      // Result of an identifier lookup in the module
                CARGBIND,    // Result of binding cargs to a lookup result
                STRUCT,      // Struct with bound cargs
                ENUM,        // Enum with bound cargs
                ENUMENTRY,   // Non-trivial enum entry access without the entry being called
                INIT,        // Init object
                NODE,        // Node object
                BLOCK,       // Block object
                EDGE,        // Edge object
                TENSOR,      // Both a forward and backward edge
                DLTYPE,      // edge or tensor type (depends on the context)
                GENERIC,     // Compile time generic type
                UNPACK       // Unpacked array
            } ty = TypeInfo::Type::INVALID;

            enum class Category
            {
                DEFAULT,  // responsible for the underlying object
                CONST,    // immutable reference to an object
                REF,      // mutable reference to an object
                VIRTUAL   // compile time object, doesn't exist at runtime
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

            union
            {
                TypeInfoType        type_type;
                TypeInfoArray       type_array;
                TypeInfoTuple       type_tuple;
                TypeInfoLookup      type_lookup;
                TypeInfoCargBind    type_cargbind;
                TypeInfoStruct      type_struct;
                TypeInfoEnum        type_enum;
                TypeInfoEnumEntry   type_enum_entry;
                TypeInfoDlType      type_dltype;
                TypeInfoGeneric     type_generic;
            };

            TypeInfo();
            TypeInfo(TypeInfo&& type) noexcept;
            TypeInfo& operator=(TypeInfo&& type) noexcept;
            ~TypeInfo();

            bool check_pos() const;
            bool check_neg() const;

            bool check_add() const;
            bool check_sub() const;
            bool check_mul() const;
            bool check_div() const;
            bool check_mod() const;
            bool check_pow() const;

            bool check_eq() const;
            bool check_ne() const;
            bool check_ge() const;
            bool check_le() const;
            bool check_gt() const;
            bool check_lt() const;

            bool check_xcfg() const;  // whether the type can be converted into a config by the interpreter
            bool check_xstr() const;  // whether the type can be converted into a string by the interpreter
            bool check_xint() const;  // whether the type can be converted into an int by the interpreter
            bool check_xflt() const;  // whether the type can be converted into a float by the interpreter
            bool check_cpy() const;
            bool check_idx() const;
            bool check_dl() const;

            std::string encode() const;
            std::string to_string() const;
            bool to_cfg(const AstNodeInfo& node_info, TypeRef& cfg) const;   // converts the underlying object into a cfg
            bool to_obj(const AstNodeInfo& node_info, TypeRef& type) const;  // converts the type into a runtime object
            bool wake_up_struct(const AstNodeInfo& node_info);  // wakes up a lazy struct as an uninitialized struct declaration
            bool wake_up_struct(const AstNodeInfo& node_info, TypeInfo::Category cat, CodegenCallback codegen);  // wakes up a lazy struct, leaving it's elements still lazy though
            template<class allowed>
            bool in_category() const
            {
                return
                    (allowed::has_default && cat == TypeInfo::Category::DEFAULT) ||
                    (allowed::has_const   && cat == TypeInfo::Category::CONST  ) ||
                    (allowed::has_ref     && cat == TypeInfo::Category::REF    ) ||
                    (allowed::has_virtual && cat == TypeInfo::Category::VIRTUAL);
            }

            // puts the object that the type references onto the top of the stack if possible
            CodegenCallback codegen = nullptr;

        private:
            void do_move(TypeInfo&& type) noexcept;
        };

        class TypeManager
        {
            friend class TypeRef;

            std::vector<TypeInfo> buf;

            inline TypeInfo* get(size_t ptr) noexcept;
            inline const TypeInfo* get(size_t ptr) const noexcept;
            inline TypeRef next();

        public:
            TypeManager();
            TypeManager(const TypeManager&) = delete;
            TypeManager(TypeManager&&) = delete;
            TypeManager& operator=(const TypeManager&) = delete;
            TypeManager& operator=(TypeManager&&) = delete;

            TypeRef duplicate          (const TypeRef src);
            TypeRef duplicate          (TypeInfo::Category cat, CodegenCallback codegen, const TypeRef src);

            TypeRef create_type        (TypeInfo::Category cat, CodegenCallback codegen, TypeRef base);
            TypeRef create_placeholder ();  // always virtual
            TypeRef create_generic     (TypeInfo::Category cat, CodegenCallback codegen, const std::string& name);
            TypeRef create_unpack      (CodegenCallback codegen, TypeRef elem);
            TypeRef create_fty         (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_bool        (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_int         (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_float       (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_string      (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_config      (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_array       (TypeInfo::Category cat, CodegenCallback codegen, TypeRef elem);
            TypeRef create_tuple       (TypeInfo::Category cat, CodegenCallback codegen, std::vector<TypeRef> elems);
            
            TypeRef create_lookup      (std::string name, CodeModule::LookupResult lookup);  // always virtual
            TypeRef create_cargbind    (std::string name, CodeModule::LookupResult lookup, const std::vector<AstExpr>& cargs); // always virtual
            TypeRef create_struct      (TypeInfo::Category cat, CodegenCallback codegen, const AstStruct* ast,
                                        const std::map<std::string, TypeRef>& cargs,
                                        const std::vector<std::pair<std::string, TypeRef>>& fields);
            TypeRef create_lazystruct  (const AstStruct* ast, const std::map<std::string, TypeRef>& cargs);  // always virtual
            TypeRef create_enum        (TypeInfo::Category cat, CodegenCallback codegen, const AstEnum* ast,
                                        const std::vector<std::string> ast_ns, const std::map<std::string, TypeRef>& cargs);
            TypeRef create_enum_entry  (const AstEnum* ast, const std::vector<std::string> ast_ns,
                                        const std::map<std::string, TypeRef>& cargs, size_t idx);

            TypeRef create_init        (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_node        (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_block       (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_edge        (TypeInfo::Category cat, CodegenCallback codegen);
            TypeRef create_tensor      (TypeInfo::Category cat, CodegenCallback codegen);

            TypeRef create_dltype      (TypeRef tensor, TypeRef edge, TypeRef shape, TypeRef fp);
        };

        bool operator==(const TypeRef& lhs, const TypeRef& rhs);
        bool operator!=(const TypeRef& lhs, const TypeRef& rhs);

        class Scope
        {
        public:
            struct StackVar
            {
                TypeRef type;
                size_t ptr;
            };

        private:
            std::map<std::string, StackVar> stack_vars;
            Scope* parent = nullptr;
            size_t curr_var_name = 0;
            int stack_size = 0;

        public:
            Scope(Scope* parent) : parent(parent) {}

            bool contains(const std::string& var_name) const;
            bool at(const std::string& var_name, StackVar& var) const;
            bool add(const std::string& var_name, TypeRef type, const AstNodeInfo& node_info);
            std::string generate_var_name();
            bool empty() const;

            bool push(size_t n=1);
            bool pop(size_t n=1);

            const Scope* get_parent() const;
            bool local_size(size_t& sz, const Scope* scope) const;
            bool list_local_vars(std::vector<StackVar>& vars, const Scope* scope);

        private:
            bool push_impl(size_t n);
            bool pop_impl(size_t n);
        };

        // Responsible for building the dependancy graph for signature cargs, vargs, and rets
        // Also for initializing the graph with types and values, and for deducing and checking the remaining elements
        class ProcCall
        {
        public:
            ProcCall(const std::vector<std::string>& sig_ns);
            ~ProcCall();

        protected:
            struct TypeNode;
            struct ValNode;

            // The value of a node can never be determined from its inputs, but the inputs can be determined by its
            struct Arg
            {
                // For stack variables (cargs and vargs), this is the name of the argument.
                // For return values, this is ret%d.  Where %d is the return value's position.
                // For temporary values, this is ~
                std::string name;
                TypeNode* type;
                bool visited = false;
                TypeRef default_type = TypeInfo::null;
            };

            struct UnaryOp
            {
                ValNode* inp = nullptr;
            };

            struct BinaryOp
            {
                ValNode* lhs = nullptr;
                ValNode* rhs = nullptr;
            };

            struct ValNode
            {
                enum class Type
                {
                    INVALID = 0,
                    ARG_VAL,
                    CONST_VAL,
                    UNARY_POS,
                    UNARY_NEG,
                    UNARY_NOT,
                    UNARY_UNPACK,
                    BINARY_ADD,
                    BINARY_SUB
                } ty = Type::INVALID;

                union
                {
                    Arg      val_arg;
                    UnaryOp  val_unary;
                    BinaryOp val_binary;
                };

                ValNode() {}
                ValNode(ValNode&& node) noexcept;
                ValNode& operator=(ValNode&& node) noexcept;
                ~ValNode();

                bool get_type(const AstNodeInfo& node_info, std::vector<TypeRef>& rets) const;

                // Start by assuming all nodes are root nodes then during ProcCall::create_* calls,
                // if the node name comes up it means that the Node wasn't a root node
                bool is_root = true;
                bool is_constref = false;  // const or ref (doesn't need to be copied before the proc call)
                const AstNodeInfo* node_info;
                TypeRef val = TypeInfo::null;  // This field gets initialized during codegen

                bool codegen_carg(Scope& scope, const std::string& name) const;
                bool codegen_varg(Scope& scope, const std::string& name) const;

            private:
                void do_move(ValNode&& node) noexcept;
            };

            struct ValType
            {
                ValNode* val = nullptr;
            };

            struct ArrayType
            {
                TypeNode* carg = nullptr;
            };

            struct TupleType
            {
                std::vector<TypeNode*> cargs;
            };

            struct StructType
            {
                const AstStruct* ast;
                std::map<std::string, ValNode*> cargs;
            };

            struct EnumType
            {
                const AstEnum* ast;
                std::vector<std::string> ast_ns;
                std::map<std::string, ValNode*> cargs;
            };

            struct DlType
            {
                ValNode* fp = nullptr;
                std::vector<ValNode*> shape;
            };

            struct TypeNode
            {
                enum class Type
                {
                    INVALID = 0,
                    TYPE,     // the type of a generic
                    INIT,
                    FTY,
                    BOOL,
                    INT,
                    FLOAT,
                    STRING,
                    CONFIG,
                    GENERIC,  // a compile time variable type
                    UNPACK,  // Unpacked types
                    ARRAY,
                    TUPLE,
                    STRUCT,
                    ENUM,
                    DLTYPE  // either a tensor or an edge depending on the context
                } ty = Type::INVALID;

                union
                {
                    ArrayType       type_array;
                    TupleType       type_tuple;
                    StructType      type_struct;
                    EnumType        type_enum;
                    DlType          type_dl;
                    ValType         type_val;
                };

                TypeNode() {}
                TypeNode(TypeNode&& node) noexcept;
                TypeNode& operator=(TypeNode&& node) noexcept;
                ~TypeNode();

                TypeRef as_type(const AstNodeInfo& node_info) const;
                bool immutable() const;  // Certain types don't allow for mutable values.  Ex. TYPE, UNPACK

                const AstNodeInfo* node_info;

            private:
                void do_move(TypeNode&& node) noexcept;
            };

            struct Carg
            {
                std::string kwarg;
                ValNode* node;
                const AstNodeInfo* node_info;
            };


            // Note: these functions are not called during codegeneration, but rather during
            // initialization.  Ex. fn foo<type T>(ref MyGenericStruct<T> x) ...
            bool type_eq(const ProcCall::TypeNode& lhs, const ProcCall::TypeNode& rhs, bool& eq);
            bool is_instance(const ValNode& val, const TypeNode& type, bool& inst);
            bool match_carg_sig(Scope& scope, const AstCargSig& sig,
                const std::vector<Carg>& param_cargs, std::map<std::string, ValNode*>& cargs);
            bool resolve_lookup(Scope& scope, const AstNodeInfo& node_info, const std::vector<Carg>& param_cargs,
                const CodeModule::LookupResult& lookup, const std::string& name, TypeNode* node);

            // Retrospective look at this system and trying to provide an explaination:
            // It seems that all the codegen calls don't actually modify the stack at all,
            // or codegen anything.  Instead, they are used to perform signature deduction
            // by updating the nodes using the passed in arguments - which is why in the
            // specific codegen functions (ex. StructCall::codegen) which do the ACTUAL
            // codegen, they only call the codegen_root_arg functions.  Additionally, they
            // don't have to pass in any data into the codegen_* functions here since the data
            // was attached to the ValNodes during the block-specific ::apply_args call.
            // So when writing or modifying any codegen function, the modifications need to
            // guarentee that they don't update the stack at all, but they DO need to act on
            // and write code that runs on the actual stack, you con't use virtual stacks.

            bool create_arg          (Scope& scope, const AstArgDecl& decl, ValNode*& node);
            bool create_arg          (Scope& scope, const AstExpr& decl, ValNode*& node);
            bool create_type         (Scope& scope, const AstExpr& expr, TypeNode*& node);
            bool create_value        (Scope& scope, const AstExpr& expr, ValNode*& node);
            bool create_value_callee (Scope& scope, const AstExpr& expr, ValNode*& node);

            bool codegen_root_arg(Scope& scope, ValNode& node);

            bool codegen_value_arg    (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_pos    (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_neg    (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_not    (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_unpack (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_add    (Scope& scope, ValNode& node, TypeRef& val);
            bool codegen_value_sub    (Scope& scope, ValNode& node, TypeRef& val);

            bool codegen_value(Scope& scope, ValNode& node, TypeRef& val);
            
            bool codegen_type_type    (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_init    (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_fty     (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_bool    (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_int     (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_float   (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_string  (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_generic (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_array   (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_tuple   (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_struct  (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_enum    (Scope& scope, TypeNode& node, TypeRef& type);
            bool codegen_type_dltype  (Scope& scope, TypeNode& node, TypeRef& type);

            bool codegen_type(Scope& scope, TypeNode& node, TypeRef& type);

            virtual TypeRef get_fp(const AstNodeInfo& node_info, TypeRef type) = 0;
            virtual TypeRef get_shape(const AstNodeInfo& node_info, TypeRef type) = 0;

            std::vector<std::string> sig_ns;
            std::map<std::string, ValNode*> carg_nodes;
            std::vector<std::string> carg_stack;
            // EDGE for intr, otherwise its TENSOR
            // Initializing the the call-specific init function
            TypeInfo::Type dltype = TypeInfo::Type::INVALID;
            const AstNodeInfo* node_info = nullptr;

            ValNode*  next_val();
            TypeNode* next_type();

        private:
            bool create_arg(
                Scope& scope,
                const AstNodeInfo& node_info,
                bool is_packed,
                const AstExpr* type_expr,
                const std::string& var_name,
                const AstExpr* default_expr,
                ValNode*& node);
            static constexpr size_t bufsz = 1024;

            ValNode val_buf[bufsz] = {};
            size_t val_buflen = 0;

            TypeNode type_buf[bufsz] = {};
            size_t type_buflen = 0;
        };

        class TensorCall :
            public ProcCall
        {
        public:
            TensorCall(const std::vector<std::string>& sig_ns);

        protected:
            virtual TypeRef get_fp(const AstNodeInfo& node_info, TypeRef type);
            virtual TypeRef get_shape(const AstNodeInfo& node_info, TypeRef type);
        };

        class EdgeCall :
            public ProcCall
        {
        public:
            EdgeCall(const std::vector<std::string>& sig_ns);

        protected:
            virtual TypeRef get_fp(const AstNodeInfo& node_info, TypeRef type);
            virtual TypeRef get_shape(const AstNodeInfo& node_info, TypeRef type);
        };

        class CommonCall :
            public ProcCall
        {
        public:
            CommonCall(const std::vector<std::string>& sig_ns);

        protected:
            virtual TypeRef get_fp(const AstNodeInfo& node_info, TypeRef type);
            virtual TypeRef get_shape(const AstNodeInfo& node_info, TypeRef type);
        };

        // Concrete call classes

        class InitCall :
            public CommonCall
        {
        public:
            using CommonCall::CommonCall;

            bool init(const AstInit& sig);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);

        private:
            std::string init_name;
            const AstNodeInfo* pinfo;
        };

        class StructCall :
            public CommonCall
        {
        public:
            using CommonCall::CommonCall;

            bool init(const AstStruct& sig);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);

        private:
            std::map<std::string, ValNode*> varg_nodes;
            std::vector<std::string> varg_stack;
            TypeNode* ret_node;
            const AstStruct* ast_struct;
        };

        class EnumCall :
            public CommonCall
        {
        public:
            using CommonCall::CommonCall;

            bool init(const AstEnum& sig, size_t idx);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);

        private:
            std::map<std::string, ValNode*> varg_nodes;
            std::vector<std::string> varg_stack;
            TypeNode* ret_node;
            const AstEnum* ast_enum;
            size_t tag_val;
        };

        class IntrCall :
            public EdgeCall
        {
        public:
            using EdgeCall::EdgeCall;

            bool init(const AstBlock& sig);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);

        private:
            std::map<std::string, ValNode*> varg_nodes;
            std::vector<std::string> varg_stack;
            std::vector<std::string> ret_stack;
            const AstBlock* ast_intr;
        };

        class DefCall :
            public TensorCall
        {
        public:
            using TensorCall::TensorCall;

            bool init(const AstBlock& sig);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);
            bool codegen_entrypoint(Scope& scope, const std::map<std::string, TypeRef>& cargs);

        private:
            std::map<std::string, ValNode*> varg_nodes;
            std::vector<std::string> varg_stack;
            std::vector<std::string> ret_stack;
            const AstBlock* ast_def;
        };

        class FnCall :
            public CommonCall
        {
        public:
            using CommonCall::CommonCall;

            bool init(const AstFn& sig);
            bool apply_args(const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs, const std::vector<TypeRef>& vargs);
            bool codegen(Scope& scope, std::vector<TypeRef>& rets);

        private:
            std::map<std::string, ValNode*> varg_nodes;
            std::vector<std::string> varg_stack;
            std::vector<TypeNode*> ret_nodes;
            const AstFn* ast_fn;
        };

        class ModuleInfo
        {
            friend bool codegen_module(ByteCodeModule& bc, ModuleInfo& info, AstModule& ast, const std::vector<std::string>& imp_dirs);

            CodeModule mod;
            AstNodeInfo node_info{ "", 0, 0, 0, 0 };

        public:
            ModuleInfo();

            void init(TypeManager* type_manager);
            bool entry_setup(const std::string& name, const std::map<std::string, TypeRef>& cargs);

            TypeRef create_bool  (BoolObj val);
            TypeRef create_fty   (FtyObj val);
            TypeRef create_int   (IntObj val);
            TypeRef create_float (FloatObj val);
            TypeRef create_str   (const StrObj& val);
            TypeRef create_array (const std::vector<TypeRef>& val);
        };

        std::string label_prefix(const AstNodeInfo& info);

        bool arg_type(TypeRef& type, Scope& scope, const AstArgDecl& arg);
        bool match_elems(const std::vector<AstExpr>& lhs, const std::vector<TypeRef>& rhs, CodegenCallback& setup_fn, std::vector<CodegenCallback>& elem_fns);
        bool match_arg(const TypeRef& arg_type, const TypeRef& param_type, std::map<std::string, TypeRef>& generics);
        bool match_carg_sig(Scope& scope, Scope& sig_scope, const std::vector<AstArgDecl>& sig, const std::vector<AstExpr>& args,
            std::map<std::string, TypeRef>& cargs, std::map<std::string, TypeRef>& generics);
        bool match_def_sig(Scope& scope, Scope& sig_scope, const AstBlock& def,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs);
        bool match_intr_sig(Scope& scope, Scope& sig_scope, const AstBlock& intr,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs);
        bool match_fn_sig(Scope& scope, Scope& sig_scope, const AstFn& fn,
            const std::vector<AstExpr>& param_cargs, const std::vector<TypeRef>& param_vargs,
            std::map<std::string, TypeRef>& cargs, std::vector<TypeRef>& vargs);
        bool resolve_lookup(Scope& scope, const AstNodeInfo& node_info, const std::string& name,
            const CodeModule::LookupResult& lookup, std::vector<TypeRef>& rets);
        bool codegen_fields(const std::string& container_type, const AstNodeInfo& node_info, const std::map<std::string, TypeRef>& cargs,
            const std::vector<AstLine>& lines, std::vector<std::pair<std::string, TypeRef>>& fields);

        bool codegen_xint(Scope& scope, const AstNodeInfo& node_info, TypeRef arg, TypeRef& cfg);
        bool codegen_xflt(Scope& scope, const AstNodeInfo& node_info, TypeRef arg, TypeRef& cfg);
        bool codegen_xstr(Scope& scope, const AstNodeInfo& node_info, TypeRef arg, TypeRef& cfg);
        bool codegen_xcfg(Scope& scope, const AstNodeInfo& node_info, TypeRef arg, TypeRef& cfg);

        template<class allowed> bool codegen_expr_single_ret(Scope& scope, const AstExpr& expr, TypeRef& ret);
        template<class allowed> bool codegen_expr_multi_ret(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        
        bool codegen_expr_callee_kw    (Scope& scope, const AstExpr& expr, TypeRef& ret);
        bool codegen_expr_callee_var   (Scope& scope, const AstExpr& expr, TypeRef& ret);
        bool codegen_expr_callee_dot   (Scope& scope, const AstExpr& expr, TypeRef& ret);
        bool codegen_expr_callee_cargs (Scope& scope, const AstExpr& expr, TypeRef& ret);

        // codegens the expr while keeping an ambiguities that may exist - used to prevent resolving lookups
        bool codegen_expr_callee(Scope& scope, const AstExpr& expr, TypeRef& ret);

        bool codegen_expr_bool     (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_int      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_float    (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_string   (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_array    (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_tuple    (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_pos      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_neg      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_not      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_unpack   (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_fwd      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_bwd      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_assign   (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_and      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_or       (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_cast     (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_idx      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_dot      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_decl     (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_cargs    (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_vargs    (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_fndecl   (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_defdecl  (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_kw       (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_expr_var      (Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);

        bool codegen_expr(Scope& scope, const AstExpr& expr, std::vector<TypeRef>& rets);
        bool codegen_exit(Scope& scope, const AstNodeInfo& info);  // unwinds and exits a scope updating the parent scope's pointers

        bool codegen_line_break    (Scope& scope, const AstLine& line);
        bool codegen_line_continue (Scope& scope, const AstLine& line);
        bool codegen_line_export   (Scope& scope, const AstLine& line);
        bool codegen_line_extern   (Scope& scope, const AstLine& line);
        bool codegen_line_return   (Scope& scope, const AstLine& line);
        bool codegen_line_intrinfo (Scope& scope, const AstLine& line);
        bool codegen_line_match    (Scope& scope, const AstLine& line);
        bool codegen_line_branch   (Scope& scope, const AstLine& line, const std::string& end_label);  // if or elif statement
        bool codegen_line_block    (Scope& scope, const AstLine& line);
        bool codegen_line_while    (Scope& scope, const AstLine& line);
        bool codegen_line_for      (Scope& scope, const AstLine& line);
        bool codegen_line_expr     (Scope& scope, const AstLine& line);

        bool codegen_line(Scope& scope, const AstLine& line);  // generates code for independent lines.  Can't handle if, elif, else, forward.
        bool codegen_lines(Scope& scope, const std::vector<AstLine>& lines);

        bool proc_name_fn     (std::string& name, const std::vector<std::string>& ns, const AstFn&     ast_fn    );
        bool proc_name_def    (std::string& name, const std::vector<std::string>& ns, const AstBlock&  ast_def   );
        bool proc_name_intr   (std::string& name, const std::vector<std::string>& ns, const AstBlock&  ast_intr  );

        bool codegen_func (const std::string& name, const AstFn& ast_fn, const std::vector<std::string>& ns);
        bool codegen_def  (const std::string& name, const AstBlock& ast_def, const std::vector<std::string>& ns);
        bool codegen_intr (const std::string& name, const AstBlock& ast_intr, const std::vector<std::string>& ns);

        bool codegen_namespace (const CodeModule::Namespace& node, std::vector<std::string>& ns);
        bool codegen_module(ByteCodeModule& bc, ModuleInfo& info, AstModule& ast, const std::vector<std::string>& imp_dirs);
    }
}

#endif
