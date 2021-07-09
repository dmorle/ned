#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include <tuple>
#include <vector>
#include <string>

#include <libnn/core/tensor.h>

namespace nn
{
    namespace nodes
    {
		class Invalid;
		class Subnode;
		class Imm;
		class Add;
		class Sub;
		class Mul;
		class Div;
		class Reshape;
		class Split;
		class Concat;
		class Einsum;
		class Conv2d;
		class Max;
		class Min;

		using node_id = uint32_t;

		template<class... _tys>
		struct _node_id_map;

		template<>
		struct _node_id_map<>
		{
			template<class T>
			static constexpr node_id _get_id(node_id id) { return 0; }

			template<class T>
			static constexpr node_id get_id() { return -1; }

			inline static constexpr size_t type_size(node_id id) { return 0; }
		};

		template<class _ty, class... _tys>
		struct _node_id_map<_ty, _tys...>
		{
			template<class T>
			static constexpr node_id _get_id(type_id id)
			{
				if (std::is_same<_ty, T>())
					return id;
				return _node_id_map<_tys...>::template _get_id<T>(id + 1);
			}

			template<class T>
			static constexpr node_id get_id() { return _node_id_map<_ty, _tys...>::template _get_id<T>(type_id()); }
		};

		using node_id_map = _node_id_map
		<
			Invalid,
			Subnode,
			Imm,
			Add,
			Sub,
			Mul,
			Div,
			Reshape,
			Split,
			Concat,
			Einsum,
			Conv2d,
			Max,
			Min
		>;
    }

    class Node
    {
	public:
        nodes::node_id id = 0;
        std::vector<std::tuple<Node&, size_t>> inps = {};
        std::vector<tensor_dsc> out_shapes = {};

		virtual void set_output_dsc(tensor_dsc& dsc, size_t out_id) = 0;
    };

	class Graph
	{
	};
}

#endif
