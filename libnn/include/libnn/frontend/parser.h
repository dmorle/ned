#ifndef NN_PARSER_H
#define NN_PARSER_H

#include <memory>

#include <libnn/core/graph.h>
#include <libnn/frontend/lexer.h>

namespace nn
{
    namespace frontend
    {
        class AstNode
        {

        };

        AstNode* parse_tokens(const TokenArray& tarr);
    }
}

#endif
