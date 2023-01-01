#ifndef NVM_COMMON_H
#define NVM_COMMON_H

#pragma warning(push, 0)

#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/BitCode/BitcodeWriter.h>

#pragma warning(pop)

#include <ned/errors.h>
#include <ned/util/libs.h>
#include <ned/core/graph.h>
#include <ned/core/config.h>
#include <ned/core/reduce.h>

#include <map>
#include <concepts>
#include <type_traits>

#define NVM_ENTRY_FN nvm_init
#define NVM_ENTRY_FN_STR "nvm_init"
#define NVM_INIT(error_fn, nodes) \
    extern "C" bool __declspec(dllexport) NVM_ENTRY_FN(::nvm::error_fn_t error_fn, ::nvm::NodeImplMap& nodes)

namespace nvm
{

    using error_fn_t = bool(*)(const std::string&);
    using Builder = llvm::IRBuilder<>;

    struct NodeCtx
    {
        nn::core::MdNodeRef node;
        const nn::core::MdGraph* graph;
    };

    class GraphCompiler;
    struct CompCtx
    {
        GraphCompiler* comp;
        llvm::Module* mod;
        Builder* builder;
        llvm::Function* func;
    };

    struct NodeImpl
    {
        bool(*match)(const NodeCtx&);
        bool(*compile)(const NodeCtx&, CompCtx&);
    };

    class NodeImplMap
    {
    public:
        NodeImplMap(std::map<std::string, std::vector<NodeImpl>>& node_impls);

        void insert(const std::string& name, const NodeImpl& impl);
        
    private:
        std::map<std::string, std::vector<NodeImpl>>& node_impls;
    };

    llvm::Type* get_fptype(llvm::LLVMContext& ctx, nn::core::EdgeFty fty);
    llvm::Value* get_fpval(llvm::LLVMContext& ctx, nn::core::EdgeFty fty, double val);

    void find_plugins(std::map<std::string, std::vector<NodeImpl>>& node_map, std::vector<nn::util::Library*>& libs);

    template<typename T> struct ImplBase
    { static bool match(const NodeCtx& node_ctx); };

    template<typename T>
    bool ImplBase<T>::match(const NodeCtx& node_ctx)
    {
        const auto& node = node_ctx.graph->get(node_ctx.node);

        // Checking the varg and ret counts
        if (node.inps.size() != T::vargs)
            return false;
        if (node.outs.size() != T::rets)
            return false;

        // Checking to make sure all the required cargs are there
        for (const std::pair<std::string, nn::core::ConfigType>& carg : T::cargs)
        {
            auto it = node.configs.find(std::get<0>(carg));
            if (it == node.configs.end() || carg.second != it->second.type)
                return false;
        }

        return true;
    }

}

#endif
