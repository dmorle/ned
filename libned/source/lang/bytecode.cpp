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

        ByteCodeBody::ByteCodeBody(const AstNodeInfo& type) :
            fname(type.fname), line_num(type.line_start), col_num(type.col_start), body_sz(0) {}

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
                return error::syntax(lbl, "label redefinition: %", lbl->val);
            label_map[lbl->val] = size();
            return false;
        }

        bool ByteCodeBody::add_label(const AstNodeInfo& type, const std::string& lbl)
        {
            if (label_map.contains(lbl))
                return error::compiler(type, "Internal error: label redefinition: %", lbl);
            label_map[lbl] = size();
            return false;
        }

        bool ByteCodeModule::add_type_bool(size_t& addr)
        {
            statics_strings.push_back("type bool");

            Obj obj;
            return
                heap.create_type_bool(obj) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_type_fty(size_t& addr)
        {
            statics_strings.push_back("type fty");

            Obj obj;
            return
                heap.create_type_fty(obj) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_type_int(size_t& addr)
        {
            statics_strings.push_back("type int");

            Obj obj;
            return
                heap.create_type_int(obj) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_type_float(size_t& addr)
        {
            statics_strings.push_back("type float");

            Obj obj;
            return
                heap.create_type_float(obj) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_type_str(size_t& addr)
        {
            statics_strings.push_back("type str");

            Obj obj;
            return
                heap.create_type_str(obj) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_obj_bool(size_t& addr, BoolObj val)
        {
            statics_strings.push_back(std::string("bool ") + (val ? "true" : "false"));

            Obj obj;
            return
                heap.create_obj_bool(obj, val) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_obj_fty(size_t& addr, FtyObj val)
        {
            std::string fty;
            if (core::fty_str(val, fty))
                return true;
            statics_strings.push_back(std::string("fty ") + fty);

            Obj obj;
            return
                heap.create_obj_fty(obj, val) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_obj_int(size_t& addr, IntObj val)
        {
            statics_strings.push_back(std::string("int ") + std::to_string(val));

            Obj obj;
            return
                heap.create_obj_int(obj, val) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_obj_float(size_t& addr, FloatObj val)
        {
            statics_strings.push_back(std::string("float ") + std::to_string(val));

            Obj obj;
            return
                heap.create_obj_float(obj, val) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_obj_str(size_t& addr, const StrObj& val)
        {
            statics_strings.push_back(std::string("str \"") + val + "\"");

            Obj obj;
            return
                heap.create_obj_str(obj, val) ||
                add_static_obj(obj, addr);
        }

        bool ByteCodeModule::add_block(const std::string& name, const ByteCodeBody& body)
        {
            if (procs.contains(name))
                return error::syntax(body.fname, body.line_num, body.col_num, "Conflicting procedure name '%'", name);
            procs.insert({ name, body });
            return false;
        }

        bool ByteCodeModule::has_proc(const std::string& name)
        {
            return procs.contains(name);
        }

        bool ByteCodeModule::add_static_ref(const TokenImp<TokenType::IDN>* label, size_t& addr)
        {
            statics_strings.push_back(std::string("proc ") + label->val);
            addr = statics.size();
            statics.push_back({ .ptr = 0 });
            static_proc_refs.push_back({ addr, label->val, label->fname, label->line_num, label->col_num });
            return false;
        }

        bool ByteCodeModule::add_static_ref(const AstNodeInfo& type, const std::string& label, size_t& addr)
        {
            statics_strings.push_back(std::string("proc ") + label);
            addr = statics.size();
            statics.push_back({ .ptr = 0 });
            static_proc_refs.push_back({ addr, label, type.fname, type.line_start, type.col_start });
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

        std::string ByteCodeModule::to_string() const
        {
            std::stringstream ss;
            for (const auto& [name, proc] : procs)
            {
                // Building a mapping from byte offsets to labels
                std::map<size_t, std::string> inv_labels;
                for (const auto& [lbl_name, lbl_offset] : proc.label_map)
                    inv_labels[lbl_offset] = lbl_name;

                ss << "\n.proc " << name << "\n";
                size_t offset = 0;
                for (const Instruction* ins : proc.instructions)
                {
                    ss << "    " << ins->to_string(statics_strings) << "\n";
                    offset += ins->get_size();
                    auto it = inv_labels.find(offset);
                    if (it != inv_labels.end())
                        ss << ":" << it->second << "\n";
                }
            }
            return ss.str();
        }

        bool ByteCodeModule::add_static_obj(Obj obj, size_t& addr)
        {
            addr = statics.size();
            statics.push_back(obj);
            return false;
        }

        namespace instruction
        {
            template<> std::string ins_str<InstructionType::JMP  >() { return "jmp"  ; }
            template<> std::string ins_str<InstructionType::BRT  >() { return "brt"  ; }
            template<> std::string ins_str<InstructionType::BRF  >() { return "brf"  ; }
            template<> std::string ins_str<InstructionType::NEW  >() { return "new"  ; }
            template<> std::string ins_str<InstructionType::AGG  >() { return "agg"  ; }
            template<> std::string ins_str<InstructionType::ARR  >() { return "arr"  ; }
            template<> std::string ins_str<InstructionType::ATY  >() { return "aty"  ; }
            template<> std::string ins_str<InstructionType::NUL  >() { return "nul"  ; }
            template<> std::string ins_str<InstructionType::POP  >() { return "pop"  ; }
            template<> std::string ins_str<InstructionType::DUP  >() { return "dup"  ; }
            template<> std::string ins_str<InstructionType::CPY  >() { return "cpy"  ; }
            template<> std::string ins_str<InstructionType::INST >() { return "inst" ; }
            template<> std::string ins_str<InstructionType::CALL >() { return "call" ; }
            template<> std::string ins_str<InstructionType::RET  >() { return "ret"  ; }
            template<> std::string ins_str<InstructionType::SET  >() { return "set"  ; }
            template<> std::string ins_str<InstructionType::IADD >() { return "iadd" ; }
            template<> std::string ins_str<InstructionType::ISUB >() { return "isub" ; }
            template<> std::string ins_str<InstructionType::IMUL >() { return "imul" ; }
            template<> std::string ins_str<InstructionType::IDIV >() { return "idiv" ; }
            template<> std::string ins_str<InstructionType::IMOD >() { return "imod" ; }
            template<> std::string ins_str<InstructionType::IPOW >() { return "ipow" ; }
            template<> std::string ins_str<InstructionType::ADD  >() { return "add"  ; }
            template<> std::string ins_str<InstructionType::SUB  >() { return "sub"  ; }
            template<> std::string ins_str<InstructionType::MUL  >() { return "mul"  ; }
            template<> std::string ins_str<InstructionType::DIV  >() { return "div"  ; }
            template<> std::string ins_str<InstructionType::MOD  >() { return "mod"  ; }
            template<> std::string ins_str<InstructionType::POW  >() { return "pow"  ; }
            template<> std::string ins_str<InstructionType::NEG  >() { return "neg"  ; }
            template<> std::string ins_str<InstructionType::LNOT >() { return "lnot" ; }
            template<> std::string ins_str<InstructionType::LAND >() { return "land" ; }
            template<> std::string ins_str<InstructionType::LOR  >() { return "lor"  ; }
            template<> std::string ins_str<InstructionType::EQ   >() { return "eq"   ; }
            template<> std::string ins_str<InstructionType::NE   >() { return "ne"   ; }
            template<> std::string ins_str<InstructionType::GT   >() { return "gt"   ; }
            template<> std::string ins_str<InstructionType::LT   >() { return "lt"   ; }
            template<> std::string ins_str<InstructionType::GE   >() { return "ge"   ; }
            template<> std::string ins_str<InstructionType::LE   >() { return "le"   ; }
            template<> std::string ins_str<InstructionType::IDX  >() { return "idx"  ; }
            template<> std::string ins_str<InstructionType::LEN  >() { return "len"  ; }
            template<> std::string ins_str<InstructionType::XSTR >() { return "xstr" ; }
            template<> std::string ins_str<InstructionType::XFLT >() { return "xflt" ; }
            template<> std::string ins_str<InstructionType::XINT >() { return "xint" ; }
            template<> std::string ins_str<InstructionType::DSP  >() { return "dsp"  ; }
            template<> std::string ins_str<InstructionType::ERR  >() { return "err"  ; }
            template<> std::string ins_str<InstructionType::EDG  >() { return "edg"  ; }
            template<> std::string ins_str<InstructionType::TSR  >() { return "tsr"  ; }
            template<> std::string ins_str<InstructionType::NDE  >() { return "nde"  ; }
            template<> std::string ins_str<InstructionType::INI  >() { return "ini"  ; }
            template<> std::string ins_str<InstructionType::BLK  >() { return "blk"  ; }
            template<> std::string ins_str<InstructionType::GFWD >() { return "gfwd" ; }
            template<> std::string ins_str<InstructionType::GBWD >() { return "gbwd" ; }
            template<> std::string ins_str<InstructionType::GINI >() { return "gini" ; }
            template<> std::string ins_str<InstructionType::SFWD >() { return "sfwd" ; }
            template<> std::string ins_str<InstructionType::SBWD >() { return "sbwd" ; }
            template<> std::string ins_str<InstructionType::SINI >() { return "sini" ; }
            template<> std::string ins_str<InstructionType::MRG  >() { return "mrg"  ; }
            template<> std::string ins_str<InstructionType::TSHP >() { return "tshp" ; }
            template<> std::string ins_str<InstructionType::TFTY >() { return "tfty" ; }
            template<> std::string ins_str<InstructionType::ESHP >() { return "eshp" ; }
            template<> std::string ins_str<InstructionType::EFTY >() { return "efty" ; }
            template<> std::string ins_str<InstructionType::EINP >() { return "einp" ; }
            template<> std::string ins_str<InstructionType::NDCFG>() { return "ndcfg"; }
            template<> std::string ins_str<InstructionType::BKCFG>() { return "bkcfg"; }
            template<> std::string ins_str<InstructionType::INCFG>() { return "incfg"; }
            template<> std::string ins_str<InstructionType::NDPRT>() { return "ndprt"; }
            template<> std::string ins_str<InstructionType::NDINP>() { return "ndinp"; }
            template<> std::string ins_str<InstructionType::NDOUT>() { return "ndout"; }
            template<> std::string ins_str<InstructionType::BKPRT>() { return "bkprt"; }
            template<> std::string ins_str<InstructionType::BKINP>() { return "bkinp"; }
            template<> std::string ins_str<InstructionType::BKOUT>() { return "bkout"; }
            template<> std::string ins_str<InstructionType::BKEXT>() { return "bkext"; }
            template<> std::string ins_str<InstructionType::BKEXP>() { return "bkexp"; }
            template<> std::string ins_str<InstructionType::PSHMD>() { return "pshmd"; }
            template<> std::string ins_str<InstructionType::POPMD>() { return "popmd"; }

            template<InstructionType OPCODE> std::string Valued<OPCODE>::
                to_string(const std::vector<std::string>& statics_strings) const { return ins_str<OPCODE>() + " " + std::to_string(val); }
            template<> std::string Valued<InstructionType::NEW>::
                to_string(const std::vector<std::string>& statics_strings) const { return "new " + statics_strings[val]; }
        }

        bool parsebc_static(const TokenArray& tarr, ByteCodeModule& mod, size_t& addr)
        {
            switch (tarr[0]->ty)
            {
            case TokenType::KW_TYPE:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static type");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_BOOL:
                    return mod.add_type_bool(addr);
                case TokenType::KW_FTY:
                    return mod.add_type_fty(addr);
                case TokenType::KW_INT:
                    return mod.add_type_int(addr);
                case TokenType::KW_FLOAT:
                    return mod.add_type_float(addr);
                case TokenType::KW_STR:
                    return mod.add_type_str(addr);
                }
                return error::syntax(tarr[0], "Invalid static type '%'", to_string(tarr[1]));
            case TokenType::KW_BOOL:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static bool");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_TRUE:
                    return mod.add_obj_bool(addr, true);
                case TokenType::KW_FALSE:
                    return mod.add_obj_bool(addr, false);
                }
                return error::syntax(tarr[0], "Invalid static bool '%'", to_string(tarr[1]));
            case TokenType::KW_FTY:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static fp");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_F16:
                    return mod.add_obj_fty(addr, core::EdgeFty::F16);
                case TokenType::KW_F32:
                    return mod.add_obj_fty(addr, core::EdgeFty::F32);
                case TokenType::KW_F64:
                    return mod.add_obj_fty(addr, core::EdgeFty::F64);
                }
                return error::syntax(tarr[0], "Invalid static fp '%'", to_string(tarr[1]));
            case TokenType::KW_INT:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static int");
                if (tarr[1]->expect<TokenType::LIT_INT>())
                    return true;
                return mod.add_obj_int(addr, tarr[1]->get<TokenType::LIT_INT>().val);
            case TokenType::KW_FLOAT:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static float");
                if (tarr[1]->expect<TokenType::LIT_FLOAT>())
                    return true;
                return mod.add_obj_float(addr, tarr[1]->get<TokenType::LIT_FLOAT>().val);
            case TokenType::KW_STR:
                if (tarr.size() != 2)
                    return error::syntax(tarr[0], "Malformed static string");
                if (tarr[1]->expect<TokenType::LIT_STR>())
                    return true;
                return mod.add_obj_str(addr, tarr[1]->get<TokenType::LIT_STR>().val);
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
            case hash("nul"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Nul(tarr[0]));
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
            case hash("ipow"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(IPow(tarr[0]));
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
            case hash("pow"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Pow(tarr[0]));
            case hash("neg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Neg(tarr[0]));
            case hash("lnot"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(LNot(tarr[0]));
            case hash("land"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(LAnd(tarr[0]));
            case hash("lor"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(LOr(tarr[0]));
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
            case hash("len"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Len(tarr[0]));
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
            case hash("err"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Err(tarr[0]));

            case hash("edg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Edg(tarr[0]));
            case hash("tsr"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Tsr(tarr[0]));
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
            case hash("gfwd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(GFwd(tarr[0]));
            case hash("gbwd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(GBwd(tarr[0]));
            case hash("gini"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(GIni(tarr[0]));
            case hash("sfwd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(SFwd(tarr[0]));
            case hash("sbwd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(SBwd(tarr[0]));
            case hash("sini"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(SIni(tarr[0]));
            case hash("mrg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Mrg(tarr[0]));
            case hash("tshp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Tshp(tarr[0]));
            case hash("tfty"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Tfty(tarr[0]));
            case hash("eshp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Eshp(tarr[0]));
            case hash("efty"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Efty(tarr[0]));
            case hash("einp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(Einp(tarr[0]));
            case hash("ndcfg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdCfg(tarr[0]));
            case hash("bkcfg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkCfg(tarr[0]));
            case hash("incfg"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(InCfg(tarr[0]));
            case hash("ndprt"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdPrt(tarr[0]));
            case hash("ndinp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdInp(tarr[0]));
            case hash("ndout"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(NdOut(tarr[0]));
            case hash("bkprt"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkPrt(tarr[0]));
            case hash("bkinp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkInp(tarr[0]));
            case hash("bkout"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkOut(tarr[0]));
            case hash("bkext"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkExt(tarr[0]));
            case hash("bkexp"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(BkExp(tarr[0]));
            case hash("pshmd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(PshMd(tarr[0]));
            case hash("popmd"):
                if (tarr.size() != 1)
                    return error::syntax(tarr[0], "Invalid instruction");
                return body.add_instruction(PopMd(tarr[0]));

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
