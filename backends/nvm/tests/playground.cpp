#include <ned/util/libs.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <functional>

using namespace nn::util;

template<typename T>
void print_vec(T* vec, size_t sz)
{
    for (size_t i = 0; i < sz - 1; i++)
        std::cout << vec[i] << ", ";
    std::cout << vec[sz - 1] << std::endl;
}

template<typename T>
void print_mrx(T* vec, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n - 1; j++)
        {
            T e = vec[i * n + j];
            std::cout << e << ", ";
        }
        std::cout << vec[i * n + n - 1] << std::endl;
    }
}

bool run_getset(Library* lib)
{
    std::function<void(uint8_t*)> getter;
    if (lib_load_symbol(lib, "getter", getter))
        return true;
    std::function<void(uint8_t*)> setter;
    if (lib_load_symbol(lib, "setter", setter))
        return true;

    float inp[4] = { 1, 2, 3, 4 };
    float out[4] = {};

    setter((uint8_t*)inp);
    getter((uint8_t*)out);

    print_vec(inp, 4);
    print_vec(out, 4);
}

bool run_matmulk0(Library* lib)
{
    std::function<void(float*)> matmulk0;
    if (lib_load_symbol(lib, "matmulk0", matmulk0))
        return true;

    float mrx[] = {1, 1, 1, 1};
    matmulk0(mrx);
    print_mrx(mrx, 2, 2);
    return false;
}

int main()
{
    if (system("llc -filetype=obj -opaque-pointers " TESTS_DIR "playground.ll") ||
        system("lld-link /dll /noentry /out:" TESTS_DIR "playground.dll " TESTS_DIR "playground.obj")
        ) return 1;

    Library* lib = nullptr;
    if (lib_new(lib, TESTS_DIR "playground.dll"))
    {
        printf("Unable to load playground.dll");
        return 1;
    }

    bool ret = run_matmulk0(lib);

    lib_del(lib);
    return ret ? 1 : 0;
}
