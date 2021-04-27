#ifndef NN_TENSOR_H
#define NN_TENSOR_H

#include <array>
#include <tuple>
#include <cstdint>

namespace nn
{
	namespace impl
	{
		template<size_t... _dims>
		struct shape;

		template<>
		struct shape<>
		{
		public:
			static constexpr size_t Rank() noexcept { return 0; }
			static constexpr std::array<size_t, 0> Shape() noexcept { return {}; }
		};

		template<size_t _dim, size_t... _dims>
		struct shape<_dim, _dims...>
		{
		public:
			static constexpr size_t Rank() noexcept { return 1 + shape<_dims...>::Rank(); }
			static constexpr std::array<size_t, shape<_dim, _dims...>::Rank()> Shape() noexcept
			{
				constexpr size_t rk = shape<_dim, _dims...>::Rank();
				std::array<size_t, rk> s{};
				size_t index = 0;
				s[index] = _dim;
				for (auto e : shape<_dims...>::Shape())
					s[++index] = e;
				return s;
			}
		};
	}

	template<typename _ty, size_t... _dims>
	class Tensor
	{
	public:
		Tensor() {}

		static constexpr size_t Size() noexcept
		{
			auto shape = Tensor<_ty, _dims...>::Shape();
			size_t size = 1;
			for (auto e : shape)
				size *= e;
			return size;
		}

		static constexpr size_t Rank() noexcept { return impl::shape<_dims...>::Rank(); }
		static constexpr std::array<size_t, Tensor<_ty, _dims...>::Rank()> Shape() noexcept {
			return impl::shape<_dims...>::Shape(); }

	protected:
		_ty _data[Tensor<_ty, _dims...>::Size()];
	};
}

#endif
