#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include <map>
#include <tuple>
#include <vector>
#include <string>

#include <libnn/core/tensor.h>

namespace nn
{
    struct Node;

    struct Edge
    {
        tensor_dsc dsc = tensor_dsc{};
        Node* input = nullptr;
        int inpid = -1;
        std::vector<std::pair<Node*, int>> outputs = {};
    };

    struct Node
    {
        std::string name;
        std::vector<Edge*> inputs;
        std::vector<Edge*> outputs;
    };

	class Graph
	{
    public:
        std::map<std::string, Edge*> inputs;
        std::map<std::string, Edge*> outputs;
	};
}

#endif
