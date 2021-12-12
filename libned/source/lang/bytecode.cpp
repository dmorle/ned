#include <ned/lang/bytecode.h>

namespace nn
{
    namespace lang
    {
        size_t CodeBlock::size() const
        {
            size_t sz = 0;
            for (const auto& e : instructions)
                sz += e->size();
            return sz;
        }

        void* CodeBlock::to_bytes(void* buf, size_t block_offset) const
        {
            size_t* inst_offsets = (size_t*)alloca(sizeof(size_t) * instructions.size());
            inst_offsets[0] = block_offset;
            for (size_t i = 0; i < instructions.size() - 1; i++)
                inst_offsets[i + 1] = inst_offsets[i] + instructions[i]->size();
            std::map<std::string, size_t> abs_label_map;
            for (auto& [key, val] : label_map)
                abs_label_map[key] = inst_offsets[val];
            
            for (const auto& e : instructions)
            {
                e->set_labels(abs_label_map);
                buf = e->to_bytes(buf);
            }
            return buf;
        }

        void CodeBlock::set_var_name(const std::string& var)
        {
            var_map[var] = 0;
        }

        void CodeBlock::add_label(const std::string& var)
        {
            label_map[var] = instructions.size();
        }

        namespace instruction
        {
            size_t Labeled::size() const
            {
                return sizeof(InstructionTypes) + sizeof(size_t);
            }

            void Labeled::set_labels(const std::map<std::string, size_t>& label_map)
            {
                label_idx = label_map.at(label);
            }
        }
    }
}
