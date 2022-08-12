#ifndef NPTX_GRAPH_OPS_H
#define NPTX_GRAPH_OPS_H

#define LLVM_WARNINGS 4267 4244 4624
#pragma warning( push )
#pragma warning( disable : LLVM_WARNINGS )

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Target/TargetMachine.h>

#pragma warning( pop )

#include <ned/core/reduce.h>

namespace npx
{
	class Op
	{
	public:
		virtual size_t nthreads() = 0;

		bool run_hardware_test(llvm::LLVMContext& ctx);

	protected:
		virtual void compile_hardware_test(llvm::LLVMContext& ctx) = 0;

		void generate_caller(llvm::IRBuilder<>& builder, llvm::Module& mod);
	};

	namespace ops
	{
		class PWAdd :
			public Op
		{
		public:
			PWAdd(nn::core::EdgeFty fty, size_t nelem);

			virtual size_t nthreads() override;

		protected:
			virtual void compile_hardware_test(llvm::LLVMContext& ctx) override;

		private:
			nn::core::EdgeFty fty;
			size_t nelem;
		};
	}
}

#endif
