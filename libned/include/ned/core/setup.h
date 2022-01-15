#ifndef NED_CORE_SETUP_H
#define NED_CORE_SETUP_H

#include <ned/core/config.h>

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace nn
{
	namespace core
	{
		enum class InitGenResult
		{
			SUCCESS,  // indicates everything worked
			INVALID,  // indicates the wrong function was called
			FAILURE   // indicates the correct function was called, but weight initialization failed
		};

		struct InitData
		{
			std::string name;
			std::map<std::string, std::unique_ptr<Config>> configs;
		};

		void load_inits(std::vector<std::string> search_paths);
		void* generate_weight(InitData& init, EdgeInfo& info);
	}
}

#endif
