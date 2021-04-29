#include <cstdio>
#include <vector>
#include <iostream>
#include <libnn/core/tensor.h>

int main()
{
	auto ten = nn::Tensor<float, 2, 2, 3, 2>();
	for (auto e : ten.Shape())
		std::cout << e << std::endl;
}
