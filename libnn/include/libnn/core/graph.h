#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include <map>
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
        std::map<std::string, Edge*> inputs;
        std::map<std::string, Edge*> outputs;
	};
}

#endif
