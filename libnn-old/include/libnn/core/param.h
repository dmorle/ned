#ifndef NN_PARAM_H
#define NN_PARAM_H

#include <libnn/core/tensor.h>

namespace nn
{
	template<typename _ty, size_t... _dims>
	class Param
	{
	public:
		Param()
			: weights(), grads()
		{}

		zero_grad()
		{

		}

		Tensor<_ty, _dims...> val;
		Tensor<_ty, _dims...> grad;
	}
}

#endif
