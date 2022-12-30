#ifndef NVM_BASIC_INTERFACE_H
#define NVM_BASIC_INTERFACE_H

#include <nvm/common.h>
#include <nvm/graphgen.h>
#include <ned/core/config.h>

#include <array>
#include <tuple>

extern nvm::error_fn_t error;

NVM_INIT(error_fn, nodes);

class AddImpl :
	public nvm::ImplBase<AddImpl>
{
public:
	static bool compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx);

	static const std::vector<std::pair<std::string, nn::core::ConfigType>> cargs;
	static constexpr size_t varg_inps = 2;
	static constexpr char name[] = "__add__";
};

#endif
