#include <cuned/cugraph.h>
#include <cassert>

#include <cuda_runtime.h>

namespace nn
{
    namespace cuda
    {
        Edge::Edge()
        {
            data = nullptr;
            id = RunId{};
            dependancy = nullptr;
        }

        Edge::~Edge()
        {
            if (data)
                cudaFree(data);
            if (dependancy)
                delete dependancy;
        }

        void* Edge::get_data(RunId id)
        {
            if (data && this->id == id)
                return data;
            assert(dependancy);
            dependancy->eval(id);
            assert(data);
            assert(this->id == id);
            return data;
        }
    }
}
