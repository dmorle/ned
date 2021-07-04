#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include <vector>
#include <memory>

#include <libnn/core/param.h>

namespace nn
{
	namespace linear_impl
	{
		template<typename _ty>
		void forward(size_t M, size_t N, const _ty* weights, const _ty* inp, _ty* out);
	}

	template<typename _ty, size_t M, size_t N>
	class Linear
	{
	public:
		Linear(Param<_ty, M, N>& params): params(params), pLastInp(nullptr) {}

		static void forward(const Tensor<_ty, M>& inp, Tensor<_ty, N>& out) const
		{
			// recording the input to be used for backprop
			pLastInp = &inp;
			linear_impl::forward(M, N, params.val._data, inp._data, out._data);
		}

		// kicking the memory can down the road...
		static void backward(const Tensor<_ty, N> ginp, Tensor<_ty, M> gout);
		static void zero_grad();

	private:
		const Tensor<_ty, M>* pLastInp;
		Param<_ty, M, N>& params
	};
}

#endif
