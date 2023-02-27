#ifndef NVM_GRAPHGEN_H
#define NVM_GRAPHGEN_H

#include <nvm/nodegen.h>

namespace nvm
{

    struct StaticData
    {
        enum class Type
        {
            INPUT,
            OUTPUT,
            EXPORT,
            EXTERN
        } ty;

        std::string name = "";
    };

    struct EdgeData  // POD
    {
        bool is_static = false;
        StaticData static_data;
        llvm::GlobalVariable* var = nullptr;
        size_t n_locks = 0;  // Nodes lock their input edges
    };

    void mark_inp_edge(EdgeData& edge_data, const std::string& name);
    void mark_out_edge(EdgeData& edge_data, const std::string& name);
    void mark_exp_edge(EdgeData& edge_data, const std::string& name);
    void mark_ext_edge(EdgeData& edge_data, const std::string& name);

    struct DepInfo
    {
        enum class Type
        {
            SYNC,
            INPUT,
            OUTPUT,
            EXPORT,
            EXTERN
        } ty;

        std::string name;

        static DepInfo from_static_data(const StaticData& static_edge_data);
    };

    class GraphCompiler
    {
    public:
        GraphCompiler();

        bool init(nn::core::MdGraph& graph);

        void print();
        bool compile();

        const EdgeData& inp_edge(const nn::core::MdNode& node, size_t idx) const;
        const EdgeData& out_edge(const nn::core::MdNode& node, size_t idx) const;

    private:
        bool init_edge(nn::core::MdEdgeRef edge);
        bool init_node(nn::core::MdNodeRef node);

        void compile_edge(nn::core::MdEdgeRef edge);
        bool compile_sync_node(
            nn::core::MdNodeRef node,
            std::vector<llvm::Function*>& funcs,
            std::vector<DepInfo>& sync_deps,
            size_t sync_id);
        bool compile_normal_node(
            nn::core::MdNodeRef node,
            std::vector<llvm::Function*>& funcs,
            std::vector<DepInfo>& sync_deps,
            size_t sync_id);
        void compile_edge_io(
            const std::map<std::string, nn::core::MdEdgeRef>& edges,
            const std::string& name);
        void compile_run_sync(const std::map<std::string, llvm::Function*>& sync_funcs);
        llvm::Function* compile_sync_fn(
            const std::string& name,
            const std::vector<llvm::Function*>& funcs,
            const std::vector<nn::core::MdNodeRef>& node_refs);
        bool compile_dll();
        bool assert_sync_node_sig(const nn::core::MdNode& node);
        
        std::pair<llvm::ArrayType*, llvm::Constant*> get_litstr(const std::string& str);
        std::string get_unique_global_name(const std::string& prefix);
        EdgeData& get_edge_data(nn::core::MdEdgeRef edge);

        static constexpr char sync_node_name[] = "sync";

        llvm::LLVMContext ctx = llvm::LLVMContext();
        llvm::Module mod = llvm::Module("mod", ctx);
        llvm::IRBuilder<> builder = llvm::IRBuilder<>(ctx);
        nn::core::MdGraph* pgraph = nullptr;

        std::vector<EdgeData> edge_data = { {} };  // null element at position 0
        std::map<std::string, size_t> node_name_counts;
        std::vector<std::string> sync_names = { "" };

        // Runtime helper functions
        llvm::Function* compile_memcpy();
        llvm::Function* memcpy = nullptr;
        llvm::Function* compile_streq();
        llvm::Function* streq = nullptr;
    };
}

#endif
