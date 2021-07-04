#include <libnn/ops/linear.h>

__global__
void cu_linear_forward(size_t M, const float* weights, const float* inp, float* out)
{
	size_t n = blockIdx.x * blockDim.x + threadIdx.x;
	out[n] = 0;
	for (size_t i = 0; i < M; i++)
		out[n] += weights[n * M + i] * inp[i];
}

void nn::linear_impl::forward(size_t M, size_t N, const float* weights, const float* inp, float* out)
{
	// bad implementation for now, will improve it later with less malloc/frees
	float* cu_weights;
	float* cu_inp;
	float* cu_out;
	cudaMalloc(&cu_weights, M * N * sizeof(float));
	cudaMalloc(&cu_inp, M * sizeof(float));
	cudaMalloc(&cu_out, N * sizeof(float));
	cudaMemcpy(cu_weights, weights, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cu_inp, inp, M * sizeof(float), cudaMemcpyHostToDevice);

	int blocknum = 256;
	cu_linear_forward<<<(N + blocknum - 1) / blocknum, blocknum >>>(M, cu_weights, cu_inp, cu_out);

	cudaMemcpy(out, cu_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(cu_weights);
	cudaFree(cu_inp);
	cudaFree(cu_out);
}
