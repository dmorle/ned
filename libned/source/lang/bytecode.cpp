#include <ned/errors.h>
#include <ned/lang/bytecode.h>

#include <vector>
#include <string>
#include <cstdio>
#include <cassert>

#define FNV_PRIME 0x00000100000001B3ULL
#define FNV_OFFSET_BASIS 0XCBF29CE484222325ULL

constexpr size_t hash(const char* s)
{
    size_t h = FNV_OFFSET_BASIS;
    for (const char* c = s; *c; c++)
        h = (h * FNV_PRIME) ^ *c;
    return h;
}

constexpr size_t hash(const std::string& s)
{
    return hash(s.c_str());
}

namespace nn
{
    namespace lang
    {
        const ByteCodeDebugInfo::Record& ByteCodeDebugInfo::at(size_t pc) const
        {
            size_t min = 0;
            size_t max = instruction_records.size();
            while (true)
            {
                size_t idx = (max + min) / 2;
                size_t idx_pc = instruction_records[idx].addr;
                if (pc < idx_pc)
                    max = idx;
                else if (idx_pc < pc)
                    min = idx;
                else
                    return instruction_records[idx];
            }
        }

        ByteCodeDebugInfo::Record Instruction::create_debug_record(size_t addr) const
        {
            return ByteCodeDebugInfo::Record{
                .addr = addr,
                .fname = fname,
                .line_num = line_num,
                .col_num = col_num
            };
        }

        ByteCodeBody::ByteCodeBody(const Token* ptk) :
            fname(ptk->fname), line_num(ptk->line_num), col_num(ptk->col_num), body_sz(0) {}

        ByteCodeBody::ByteCodeBody(const AstNodeInfo& info) :
            fname(info.fname), line_num(info.line_start), col_num(info.col_start), body_sz(0) {}

        size_t ByteCodeBody::size() const
        {
            return body_sz;
        }

        CodeSegPtr ByteCodeBody::to_bytes(size_t offset, CodeSegPtr base, ByteCodeDebugInfo& debug_info) const
        {
            std::map<std::string, size_t> abs_label_map;
            for (auto& [key, val] : label_map)
                abs_label_map[key] = offset + val;
            
            CodeSegPtr buf = base;
            for (const auto& e : instructions)
            {
                if (e->set_labels(abs_label_map))
                    return nullptr;
                buf = e->to_bytes(buf);
                debug_info.instruction_records.push_back(std::move(e->create_debug_record(offset + (buf - base))));
            }
            return buf;
        }

        bool ByteCodeBody::add_label(const TokenImp<TokenType::IDN>* lbl)
        {
            if (label_map.contains(lbl->val))
                return error::syntax(lbl, "Label redefinition");
            label_map[lbl->val] = size();
            return false;
        }

        bool ByteCodeModule::add_block(const std::string& name, const ByteCodeBody& body)
        {
            if (procs.contains(name))
                return error::syntax(body.fname, body.line_num, body.col_num, "Conflicting procedure name '%'", name);
            procs.insert({ name, body });
            return false;
        }

        bool ByteCodeModule::add_static_obj(Obj obj, size_t& addr)
        {
            addr = statics.size();
            statics.push_back(obj);
            return false;
        }

        bool ByteCodeModule::add_static_ref(const TokenImp<TokenType::IDN>* label, size_t& addr)
        {
            addr = statics.size();
            statics.push_back({ .ptr = 0 });
            static_proc_refs.push_back({ addr, label->val, label->fname, label->line_num, label->col_num });
            return false;
        }

        bool ByteCodeModule::add_static_ref(const std::string& label, const std::string& fname, size_t line_num, size_t col_num, size_t& addr)
        {
            addr = statics.size();
            statics.push_back({ .ptr = 0 });
            static_proc_refs.push_back({ addr, label, fname, line_num, col_num });
            return false;
        }

        bool ByteCodeModule::export_module(ByteCode& byte_code)
        {
            // Ordering the procs and getting the offsets
            std::vector<std::string> block_order;
            size_t code_segment_sz = 0;
            for (const auto& [name, body] : procs)
            {
                block_order.push_back(name);
                byte_code.proc_offsets[name] = code_segment_sz;
                code_segment_sz += body.size();
            }

            // Replacing the static references
            for (const auto& [addr, label, fname, line_num, col_num] : static_proc_refs)
            {
                if (!procs.contains(label))
                    return error::syntax(fname, line_num, col_num, "Reference to undefined procedure '%'", label);
                assert(0 <= addr && addr < statics.size());
                statics[addr].ptr = byte_code.proc_offsets.at(label);
            }

            // Constructing the code segment
            byte_code.code_segment = (CodeSegPtr)std::malloc(code_segment_sz);
            if (!byte_code.code_segment)
                throw std::bad_alloc();  // I'll be getting rid of exceptions when I get around to creating my own data structures
            CodeSegPtr buf = byte_code.code_segment;
            for (const auto& name : block_order)
            {
                buf = procs.at(name).to_bytes(byte_code.proc_offsets.at(name), buf, byte_code.debug_info);
                if (!buf)
                    return true;
            }
            assert(buf - byte_code.code_segment == code_segment_sz);

            // Constructing the data segment
            byte_code.data_segment = (DataSegPtr)std::malloc(sizeof(Obj) * statics.size());
            memcpy(byte_code.data_segment, statics.data(), sizeof(Obj) * statics.size());
            return false;
        }

        bool parsebc_static(const TokenArray& tarr, ByteCodeModule& mod, size_t& addr)
        {
            Obj obj;
            switch (tarr[0]->ty)
            {
            case TokenType::KW_TYPE:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static type");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_BOOL:
                    return
                        mod.heap.create_type_bool(obj) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_FP:
                    return
                        mod.heap.create_type_fty(obj) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_INT:
                    return
                        mod.heap.create_type_int(obj) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_FLOAT:
                    return
                        mod.heap.create_type_float(obj) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_STR:
                    return
                        mod.heap.create_type_str(obj) ||
                        mod.add_static_obj(obj, addr);
                }
                return error::syntax(tarr[0], "Invalid static type '%'", to_string(tarr[1]));
            case TokenType::KW_BOOL:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static bool");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_TRUE:
                    return
                        mod.heap.create_obj_bool(obj, true) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_FALSE:
                    return
                        mod.heap.create_obj_bool(obj, false) ||
                        mod.add_static_obj(obj, addr);
                }
                return error::syntax(tarr[0], "Invalid static bool '%'", to_string(tarr[1]));
            case TokenType::KW_FP:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static fp");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_F16:
                    return
                        mod.heap.create_obj_fwidth(obj, core::EdgeFty::F16) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_F32:
                    return
                        mod.heap.create_obj_fwidth(obj, core::EdgeFty::F32) ||
                        mod.add_static_obj(obj, addr);
                case TokenType::KW_F64:
                    return
                        mod.heap.create_obj_fwidth(obj, core::EdgeFty::F64) ||
                        mod.add_static_obj(obj, addr);
                }
                return error::syntax(tarr[0], "Invalid static fp '%'", to_string(tarr[1]));
            case TokenType::KW_INT:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static int");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                return
                    mod.heap.create_obj_int(obj, tarr[1]->get<TokenType::LIT_INT>().val) ||
                    mod.add_static_obj(obj, addr);
            case TokenType::KW_FLOAT:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static float");
                if (tarr[1]->expect<TokenType::LIT_FLOAT>())
                    return true;
                return
                    mod.heap.create_obj_float(obj, tarr[1]->get<TokenType::LIT_FLOAT>().val) ||
                    mod.add_static_obj(obj, addr);
            case TokenType::KW_STR:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static string");
                if (tarr[1]->expect<TokenType::LIT_STR>())
                    return true;
                return
                    mod.heap.create_obj_str(obj, tarr[1]->get<TokenType::LIT_STR>().val) ||
                    mod.add_static_obj(obj, addr);
            case TokenType::IDN:
                if (std::string(tarr[0]->get<TokenType::IDN>().val) != "proc")
                    break;
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static proc");
                if (tarr[1]->expect<TokenType::IDN>())
                    return true;
                return mod.add_static_ref(&tarr[1]->get<TokenType::IDN>(), addr);
            }
            return error::syntax(tarr[0], "Invalid static type '%'", to_string(tarr[0]));
        }

        bool parsebc_instruction(const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body)
        {
            assert(tarr.size());

            using namespace instruction;
            if (tarr[0]->expect<TokenType::IDN>())
                return true;
            switch (hash(tarr[0]->get<TokenType::IDN>().val))
            {
            case hash("jmp"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>())
                    return true;
                return body.add_instruction(Jmp(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("brt"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>())
                    return true;
                return body.add_instruction(Brt(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("brf"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>())
                    return true;
                return body.add_instruction(Brf(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("new"):
            {
                if (tarr.size() != 3)
                    return error::syntax(tarr[0], "Invalid instruction");
                size_t addr;
                return
                    parsebc_static({ tarr, 1 }, mod, addr) ||
                    body.add_instruction(New(tarr[0], addr));
            }
            case hash("agg"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return error::syntax(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Agg(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("arr"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Arr(tarr[0]));
            case hash("aty"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return error::syntax(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Aty(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("pop"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return error::syntax(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Pop(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("dup"):
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return error::syntax(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Dup(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("cpy"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Cpy(tarr[0]));
            case hash("inst"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Inst(tarr[0]));
            case hash("call"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Call(tarr[0]));
            case hash("ret"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Ret(tarr[0]));
            case hash("set"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Set(tarr[0]));
            case hash("iadd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(IAdd(tarr[0]));
            case hash("isub"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(ISub(tarr[0]));
            case hash("imul"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(IMul(tarr[0]));
            case hash("idiv"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(IDiv(tarr[0]));
            case hash("imod"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(IMod(tarr[0]));
            case hash("add"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Add(tarr[0]));
            case hash("sub"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Sub(tarr[0]));
            case hash("mul"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Mul(tarr[0]));
            case hash("div"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Div(tarr[0]));
            case hash("mod"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Mod(tarr[0]));
            case hash("eq"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Eq(tarr[0]));
            case hash("ne"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Ne(tarr[0]));
            case hash("ge"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Ge(tarr[0]));
            case hash("le"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Le(tarr[0]));
            case hash("gt"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Gt(tarr[0]));
            case hash("lt"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Lt(tarr[0]));
            case hash("idx"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Idx(tarr[0]));
            case hash("xstr"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(XStr(tarr[0]));
            case hash("xflt"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(XFlt(tarr[0]));
            case hash("xint"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(XInt(tarr[0]));
            case hash("dsp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Dsp(tarr[0]));

            case hash("edg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Edg(tarr[0]));
            case hash("nde"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Nde(tarr[0]));
            case hash("ini"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Ini(tarr[0]));
            case hash("blk"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Blk(tarr[0]));
            case hash("ndinp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdInp(tarr[0]));
            case hash("ndout"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdOut(tarr[0]));
            case hash("bkinp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkInp(tarr[0]));
            case hash("bkout"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkOut(tarr[0]));
            case hash("pshmd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(PshMd(tarr[0]));
            case hash("popmd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(PopMd(tarr[0]));
            case hash("ext"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Ext(tarr[0]));
            case hash("exp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Exp(tarr[0]));

            default:
                return error::syntax(tarr[0], "Unrecognized opcode: %", tarr[0]->get<TokenType::IDN>().val);
            }
        }

        bool parsebc_body(const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body)
        {
            int i = 0;
            bool ret = false;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            while (i < tarr.size())
            {
                int end = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                if (end < 0)
                    end = tarr.size();
                if (tarr[i]->ty == TokenType::COLON)
                {
                    for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                    tarr[i]->expect<TokenType::IDN>() || body.add_label(&tarr[i]->get<TokenType::IDN>());
                }
                else
                    ret = ret || parsebc_instruction({ tarr, i, end }, mod, body);
                for (i = end; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            return ret;
        }

        bool parsebc_module(const TokenArray& tarr, ByteCodeModule& mod)
        {
            bool ret = false;
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            while (i < tarr.size())
            {
                if (tarr[i]->expect<TokenType::DOT>())
                    return true;
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                if (strcmp(tarr[i]->get<TokenType::IDN>().val, "proc"))
                    return error::syntax(tarr[i], "Expected keyword 'proc'");
                for (i++; i < tarr.size() && tarr[i]->is_whitespace(); i++);
                if (tarr[i]->expect<TokenType::IDN>())
                    return true;
                std::string name = tarr[i++]->get<TokenType::IDN>().val;
                if (tarr[i++]->expect<TokenType::ENDL>())
                    return true;
                int end = tarr.search(IsSameCriteria(TokenType::DOT), i);
                if (end < 0)
                    end = tarr.size();
                ByteCodeBody body{ tarr[i] };
                if (parsebc_body({ tarr, i, end }, mod, body))
                    ret = true;
                else
                    ret = ret || mod.add_block(name, body);
                for (i = end; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            return ret;
        }
    }
}
