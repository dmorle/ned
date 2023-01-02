#ifndef NVM_RUNTIME_H
#define NVM_RUNTIME_H

#include <ned/util/libs.h>

namespace nvm
{

	class Runtime
	{
	public:
		Runtime();
		~Runtime();

		bool init(const std::string& fname);

		void run();
		bool run_sync(const std::string& sync_name);
		template<typename T> bool get_inp(const std::string& inp_name, T* buf) {
			return get_inp_impl(inp_name, (uint8_t*)buf); }
		template<typename T> bool set_inp(const std::string& inp_name, T* buf) {
			return set_inp_impl(inp_name, (uint8_t*)buf); }
		template<typename T> bool get_out(const std::string& out_name, T* buf) {
			return get_out_impl(out_name, (uint8_t*)buf); }
		template<typename T> bool set_out(const std::string& out_name, T* buf) {
			return set_out_impl(out_name, (uint8_t*)buf); }
		template<typename T> bool get_exp(const std::string& inp_name, T* buf) {
			return get_inp_impl(inp_name, (uint8_t*)buf); }
		template<typename T> bool set_exp(const std::string& inp_name, T* buf) {
			return set_inp_impl(inp_name, (uint8_t*)buf); }
		template<typename T> bool get_ext(const std::string& out_name, T* buf) {
			return get_out_impl(out_name, (uint8_t*)buf); }
		template<typename T> bool set_ext(const std::string& out_name, T* buf) {
			return set_out_impl(out_name, (uint8_t*)buf); }

	private:
		nn::util::Library* lib = nullptr;
		std::function<void()> run_fn;
		std::function<uint32_t(const char*)> run_sync_fn;
		std::function<uint32_t(const char*, uint8_t*)> get_inp_fn;
		std::function<uint32_t(const char*, uint8_t*)> set_inp_fn;
		std::function<uint32_t(const char*, uint8_t*)> get_out_fn;
		std::function<uint32_t(const char*, uint8_t*)> set_out_fn;
		std::function<uint32_t(const char*, uint8_t*)> get_exp_fn;
		std::function<uint32_t(const char*, uint8_t*)> set_exp_fn;
		std::function<uint32_t(const char*, uint8_t*)> get_ext_fn;
		std::function<uint32_t(const char*, uint8_t*)> set_ext_fn;

		bool get_inp_impl(const std::string& inp_name, uint8_t* buf);
		bool set_inp_impl(const std::string& inp_name, uint8_t* buf);
		bool get_out_impl(const std::string& out_name, uint8_t* buf);
		bool set_out_impl(const std::string& out_name, uint8_t* buf);
		bool get_exp_impl(const std::string& inp_name, uint8_t* buf);
		bool set_exp_impl(const std::string& inp_name, uint8_t* buf);
		bool get_ext_impl(const std::string& out_name, uint8_t* buf);
		bool set_ext_impl(const std::string& out_name, uint8_t* buf);
	};

}

#endif
