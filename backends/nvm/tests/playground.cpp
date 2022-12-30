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

void on_err(Library* lib)
{
	lib_del(lib);
	exit(1);
}

int main()
{
	if (system("llc -filetype=obj " TESTS_DIR "playground.ll") ||
		system("lld-link /dll /noentry /out:" TESTS_DIR "playground.dll " TESTS_DIR "playground.obj")
		) return 1;

	Library* lib = nullptr;
	if (lib_new(lib, TESTS_DIR "playground.dll"))
	{
		printf("Unable to load playground.dll");
		return 1;
	}

	std::function<void(uint8_t*)> getter;
	if (lib_load_symbol(lib, "getter", getter))
		on_err(lib);
	std::function<void(uint8_t*)> setter;
	if (lib_load_symbol(lib, "setter", setter))
		on_err(lib);

	float inp[4] = { 1, 2, 3, 4 };
	float out[4] = {};

	setter((uint8_t*)inp);
	getter((uint8_t*)out);

	print_vec(inp, 4);
	print_vec(out, 4);

	return 0;
}
