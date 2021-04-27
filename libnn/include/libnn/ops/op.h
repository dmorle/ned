#ifndef NN_OP_H
#define NN_OP_H

#include <libnn/core/tensor.h>

typedef struct
{

	void(*forward)()
}
nn_op_st;

#endif
