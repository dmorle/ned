#include <ned/util/libs.h>

#include <string>
#include <windows.h>

namespace nn
{
	namespace util
	{
#ifdef WIN32
		struct Library
		{
			HMODULE hdll;
		};

		bool lib_new(Library*& lib, const std::string& name)
		{
			lib = new Library();
			lib->hdll = LoadLibrary(name.c_str());  // LPCSTR
			return lib->hdll == nullptr;
		}

		bool lib_del(Library* lib)
		{
			bool ret = FreeLibrary(lib->hdll);
			delete lib;
			return ret;
		}

		bool lib_load_symbol(Library* lib, const std::string& name, void*& proc)
		{
			proc = GetProcAddress(lib->hdll, name.c_str());
			return proc == nullptr;
		}
#endif
	}
}
