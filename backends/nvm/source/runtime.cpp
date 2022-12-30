#include <nvm/runtime.h>

#include <ned/errors.h>

namespace nvm
{

	using namespace nn;

	Runtime::Runtime() {}

	Runtime::~Runtime()
	{
		if (lib)
			util::lib_del(lib);
	}

	bool Runtime::init(const std::string& fname)
	{
		if (util::lib_new(lib, fname))
			return error::graph("Unable to load runtime graph %", fname);
		if (util::lib_load_symbol(lib, "step", step_fn))
			return error::graph("Unable to load step function from %", fname);
		if (util::lib_load_symbol(lib, "get_inp", get_inp_fn))
			return error::graph("Unable to load get_inp function from %", fname);
		if (util::lib_load_symbol(lib, "set_inp", *(void**)&set_inp_fn))
			return error::graph("Unable to load set_inp function from %", fname);
		if (util::lib_load_symbol(lib, "get_out", get_out_fn))
			return error::graph("Unable to load get_out function from %", fname);
		if (util::lib_load_symbol(lib, "set_out", set_out_fn))
			return error::graph("Unable to load set_out function from %", fname);
	
		return false;
	}

	void Runtime::step()
	{
		step_fn();
	}

	bool Runtime::get_inp_impl(const std::string& inp_name, uint8_t* buf)
	{
		uint32_t ret = get_inp_fn(inp_name.c_str(), buf);
		if (ret)
			return error::graph("Unable to find input %", inp_name);
		return false;
	}

	bool Runtime::set_inp_impl(const std::string& inp_name, uint8_t* buf)
	{
		const char* inp_name_cstr = inp_name.c_str();
		uint32_t ret = set_inp_fn(inp_name_cstr, buf);
		if (ret)
			return error::graph("Unable to find input %", inp_name);
		return false;
	}
	
	bool Runtime::get_out_impl(const std::string& out_name, uint8_t* buf)
	{
		uint32_t ret = get_out_fn(out_name.c_str(), buf);
		if (ret)
			return error::graph("Unable to find output %", out_name);
		return false;
	}

	bool Runtime::set_out_impl(const std::string& out_name, uint8_t* buf)
	{
		uint32_t ret = set_out_fn(out_name.c_str(), buf);
		if (ret)
			return error::graph("Unable to find output %", out_name);
		return false;
	}

}
