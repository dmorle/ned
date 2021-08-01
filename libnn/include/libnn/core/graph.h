#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include <tuple>
#include <vector>
#include <string>

#include <libnn/core/tensor.h>

namespace nn
{
    class Node;

    class Edge
    {
    public:
        bool is_static;
        std::string name;
        tensor_dsc dsc;
        Node* input;
        int inpid;
        std::vector<std::pair<Node*, int>> outputs;
    };

    class Node
    {
	public:
        std::string name;
        std::vector<Edge*> inputs;
        std::vector<Edge*> outputs;
    };

	class Graph
	{
    public:
        std::vector<Edge*> inputs;
        std::vector<Edge*> outputs;
	};
}

#endif
