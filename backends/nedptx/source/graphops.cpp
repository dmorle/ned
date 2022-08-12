#include <nedptx/graphops.h>

#include <iostream>
#include <fstream>
#include <iterator>
#include <cassert>

#include <cuda.h>

void checkCudaErrors(CUresult err) { assert(err == CUDA_SUCCESS); }

namespace npx
{
	bool Op::run_hardware_test(llvm::LLVMContext& ctx)
	{
        CUdevice    device;
        CUmodule    cudaModule;
        CUcontext   context;
        CUfunction  function;
        CUlinkState linker;
        int         devCount;

        // CUDA initialization
        checkCudaErrors(cuInit(0));
        checkCudaErrors(cuDeviceGetCount(&devCount));
        checkCudaErrors(cuDeviceGet(&device, 0));

        char name[128];
        checkCudaErrors(cuDeviceGetName(name, 128, device));

        int devMajor, devMinor;
        checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
        if (devMajor < 2)
            return true;

        std::ifstream ifs("kernel.ptx");
        if (!ifs.is_open())
            return true;

        std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

        // Create driver context
        checkCudaErrors(cuCtxCreate(&context, 0, device));

        // Create module for object
        checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));

        // Get kernel function.  This should be defined by the compiled llvm code
        checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

        // Device data
        CUdeviceptr devBufferA;
        CUdeviceptr devBufferB;
        CUdeviceptr devBufferC;

        checkCudaErrors(cuMemAlloc(&devBufferA, sizeof(float)*16));
        checkCudaErrors(cuMemAlloc(&devBufferB, sizeof(float)*16));
        checkCudaErrors(cuMemAlloc(&devBufferC, sizeof(float)*16));

        float* hostA = new float[16];
        float* hostB = new float[16];
        float* hostC = new float[16];

        // Populate input
        for (unsigned i = 0; i != 16; ++i) {
            hostA[i] = (float)i;
            hostB[i] = (float)(2*i);
            hostC[i] = 0.0f;
        }

        checkCudaErrors(cuMemcpyHtoD(devBufferA, &hostA[0], sizeof(float)*16));
        checkCudaErrors(cuMemcpyHtoD(devBufferB, &hostB[0], sizeof(float)*16));

        unsigned blockSizeX = 16;
        unsigned blockSizeY = 1;
        unsigned blockSizeZ = 1;
        unsigned gridSizeX = 1;
        unsigned gridSizeY = 1;
        unsigned gridSizeZ = 1;

        // Kernel parameters
        void* KernelParams[] = { &devBufferA, &devBufferB, &devBufferC };

        // Kernel launch.  TODO: time the kernel
        checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
            blockSizeX, blockSizeY, blockSizeZ,
            0, NULL, KernelParams, NULL));

        // Retrieve device data
        checkCudaErrors(cuMemcpyDtoH(&hostC[0], devBufferC, sizeof(float)*16));

        // Clean up after ourselves
        delete[] hostA;
        delete[] hostB;
        delete[] hostC;

        // Clean-up
        checkCudaErrors(cuMemFree(devBufferA));
        checkCudaErrors(cuMemFree(devBufferB));
        checkCudaErrors(cuMemFree(devBufferC));
        checkCudaErrors(cuModuleUnload(cudaModule));
        checkCudaErrors(cuCtxDestroy(context));

        return false;
	}

	void Op::generate_caller(llvm::IRBuilder<>& builder, llvm::Module& mod)
	{
		// External function declarations needed from the cuda driver
		llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getInt32Ty(mod.getContext()), false);
		llvm::Function* func = llvm::Function::Create(func_ty, llvm::GlobalValue::LinkageTypes::ExternalLinkage, "main", &mod);
		
		//mod.getOrInsertFunction()
	}

	namespace ops
	{
		PWAdd::PWAdd(nn::core::EdgeFty fty, size_t nelem) :
			fty(fty), nelem(nelem) {}

		void PWAdd::compile_hardware_test(llvm::LLVMContext& ctx)
		{
			auto* builder = new llvm::IRBuilder<>(ctx);
			auto* mod = new llvm::Module("kernel", ctx);
			

		}
	}
}
