#ifndef NVM_BASIC_INTERFACE_H
#define NVM_BASIC_INTERFACE_H

#include <nvm/common.h>
#include <nvm/graphgen.h>
#include <ned/core/config.h>

#include <array>
#include <tuple>

extern nvm::error_fn_t error;

NVM_INIT(error_fn, nodes);

struct AddImpl :
    public nvm::ImplBase<AddImpl>
{
    static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

    static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
    static constexpr size_t vargs = 2;
    static constexpr size_t rets = 1;
    static constexpr char name[] = "__add__";
};

struct MulImpl :
    public nvm::ImplBase<MulImpl>
{
    static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

    static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
    static constexpr size_t vargs = 2;
    static constexpr size_t rets = 1;
    static constexpr char name[] = "__mul__";
};

struct ConstValImpl :
    public nvm::ImplBase<ConstValImpl>
{
    static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

    static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
    static constexpr size_t vargs = 0;
    static constexpr size_t rets = 1;
    static constexpr char name[] = "const_val";
};

struct TransposeImpl :
    public nvm::ImplBase<TransposeImpl>
{
    static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

    static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
    static constexpr size_t vargs = 1;
    static constexpr size_t rets = 1;
    static constexpr char name[] = "transpose";
};

struct MatmulImpl :
    public nvm::ImplBase<MatmulImpl>
{
    static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

    static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
    static constexpr size_t vargs = 2;
    static constexpr size_t rets = 1;
    static constexpr char name[] = "matmul";
};

#endif
