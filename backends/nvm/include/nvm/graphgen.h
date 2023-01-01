#ifndef NVM_GRAPHGEN_H
#define NVM_GRAPHGEN_H

#include <nvm/common.h>

namespace nvm
{
    struct EdgeData  // POD
    {
        bool is_static = false;
        llvm::GlobalVariable* var = nullptr;
        size_t n_locks = 0;  // Nodes lock their input edges
    };

    class GraphCompiler
    {
    public:
        GraphCompiler();
        ~GraphCompiler();

        bool init(nn::core::MdGraph& graph);

        void print();
        bool compile();

        const EdgeData& inp_edge(const nn::core::MdNode& node, size_t idx) const;
        const EdgeData& out_edge(const nn::core::MdNode& node, size_t idx) const;

    private:
        bool init_edge(nn::core::MdEdgeRef edge);
        bool init_node(nn::core::MdNodeRef node);

        void compile_edge(nn::core::MdEdgeRef edge);
        bool compile_node(
            nn::core::MdNodeRef node,
            std::vector<llvm::Function*>& funcs,
            std::vector<std::string>& sync_deps,
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

        // Stuff needed for plugins
        std::map<std::string, std::vector<NodeImpl>> node_map;
        std::vector<nn::util::Library*> plugin_libs;
    };
}

#endif
