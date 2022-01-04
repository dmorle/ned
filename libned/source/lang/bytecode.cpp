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
        size_t ByteCodeBody::size() const
        {
            return body_sz;
        }

        CodeSegPtr ByteCodeBody::to_bytes(Errors& errs, size_t offset, CodeSegPtr buf) const
        {
            std::map<std::string, size_t> abs_label_map;
            for (auto& [key, val] : label_map)
                abs_label_map[key] = offset + val;
            
            for (const auto& e : instructions)
            {
                if (e->set_labels(errs, abs_label_map))
                    return nullptr;
                buf = e->to_bytes(buf);
            }
            return buf;
        }

        bool ByteCodeBody::add_label(Errors& errs, const TokenImp<TokenType::IDN>* lbl)
        {
            if (label_map.contains(lbl->val))
                return errs.add(lbl, "Label redefinition");
            label_map[lbl->val] = instructions.size();
        }

        bool parsebc_static(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, Obj& obj)
        {
            switch (tarr[0]->ty)
            {
            case TokenType::KW_TYPE:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static type");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_BOOL:
                    return mod.heap.create_type_bool(errs, obj);
                case TokenType::KW_FP:
                    return mod.heap.create_type_fwidth(errs, obj);
                case TokenType::KW_INT:
                    return mod.heap.create_type_int(errs, obj);
                case TokenType::KW_FLOAT:
                    return mod.heap.create_type_float(errs, obj);
                case TokenType::KW_STR:
                    return mod.heap.create_type_str(errs, obj);
                }
                return errs.add(tarr[0], "Invalid static type '{}'", to_string(tarr[1]));
            case TokenType::KW_BOOL:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static bool");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_TRUE:
                    return mod.heap.create_obj_bool(errs, obj, true);
                case TokenType::KW_FALSE:
                    return mod.heap.create_obj_bool(errs, obj, false);
                }
                return errs.add(tarr[0], "Invalid static bool '{}'", to_string(tarr[1]));
            case TokenType::KW_FP:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static fp");
                switch (tarr[1]->ty)
                {
                case TokenType::KW_F16:
                    return mod.heap.create_obj_fwidth(errs, obj, core::tensor_dty::F16);
                case TokenType::KW_F32:
                    return mod.heap.create_obj_fwidth(errs, obj, core::tensor_dty::F32);
                case TokenType::KW_F64:
                    return mod.heap.create_obj_fwidth(errs, obj, core::tensor_dty::F64);
                }
                return errs.add(tarr[0], "Invalid static fp '{}'", to_string(tarr[1]));
            case TokenType::KW_INT:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static int");
                if (tarr[1]->expect<TokenType::LIT_INT>(errs))
                    return true;
                return mod.heap.create_obj_int(errs, obj, tarr[1]->get<TokenType::LIT_INT>().val);
            case TokenType::KW_FLOAT:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static float");
                if (tarr[1]->expect<TokenType::LIT_FLOAT>(errs))
                    return true;
                return mod.heap.create_obj_float(errs, obj, tarr[1]->get<TokenType::LIT_FLOAT>().val);
            case TokenType::KW_STR:
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Malformed static string");
                if (tarr[1]->expect<TokenType::LIT_STR>(errs))
                    return true;
                return mod.heap.create_obj_str(errs, obj, tarr[1]->get<TokenType::LIT_STR>().val);
            }
            return errs.add(tarr[0], "Invalid static type '{}'", to_string(tarr[0]));
        }

        bool parsebc_instruction(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body)
        {
            assert(tarr.size());

            using namespace instruction;
            if (tarr[0]->expect<TokenType::IDN>(errs))
                return true;
            switch (hash(tarr[0]->get<TokenType::IDN>().val))
            {
            case hash("jmp"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>(errs))
                    return true;
                return body.add_instruction(Jmp(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("brt"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>(errs))
                    return true;
                return body.add_instruction(Brt(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("brf"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::IDN>(errs))
                    return true;
                return body.add_instruction(Brf(tarr[0], tarr[1]->get<TokenType::IDN>().val));
            case hash("new"):
            {
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                Obj obj;
                size_t addr;
                return
                    parsebc_static(errs, { tarr, 1 }, mod, obj) ||
                    mod.add_static(obj, addr) ||
                    body.add_instruction(New(tarr[0], addr));
            }
            case hash("agg"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>(errs))
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return errs.add(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Agg(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("arr"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Arr(tarr[0]));
            case hash("aty"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>(errs))
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return errs.add(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Aty(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("pop"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>(errs))
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return errs.add(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Pop(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("dup"):
                if (tarr.size() != 2)
                    return errs.add(tarr[0], "Invalid instruction");
                if (tarr[1]->expect<TokenType::LIT_INT>(errs))
                    return true;
                if (tarr[1]->get<TokenType::LIT_INT>().val < 0)
                    return errs.add(tarr[1], "Value type instructions require a strictly positive integer");
                return body.add_instruction(Dup(tarr[0], tarr[1]->get<TokenType::LIT_INT>().val));
            case hash("cpy"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Cpy(tarr[0]));
            case hash("inst"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Inst(tarr[0]));
            case hash("call"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Call(tarr[0]));
            case hash("ret"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Ret(tarr[0]));
            case hash("set"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Set(tarr[0]));
            case hash("iadd"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(IAdd(tarr[0]));
            case hash("isub"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(ISub(tarr[0]));
            case hash("imul"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(IMul(tarr[0]));
            case hash("idiv"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(IDiv(tarr[0]));
            case hash("imod"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(IMod(tarr[0]));
            case hash("add"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Add(tarr[0]));
            case hash("sub"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Sub(tarr[0]));
            case hash("mul"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Mul(tarr[0]));
            case hash("div"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Div(tarr[0]));
            case hash("mod"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Mod(tarr[0]));
            case hash("eq"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Eq(tarr[0]));
            case hash("ne"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Ne(tarr[0]));
            case hash("ge"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Ge(tarr[0]));
            case hash("le"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Le(tarr[0]));
            case hash("gt"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Gt(tarr[0]));
            case hash("lt"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Lt(tarr[0]));
            case hash("idx"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(Idx(tarr[0]));
            case hash("xstr"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(XStr(tarr[0]));
            case hash("xflt"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(XFlt(tarr[0]));
            case hash("xint"):
                if (tarr.size() != 1)
                    return errs.add(tarr[0], "Invalid instruction");
                return body.add_instruction(XInt(tarr[0]));
            default:
                return errs.add(tarr[0], "Unrecognized opcode: {}", tarr[0]->get<TokenType::IDN>().val);
            }
        }

        bool parsebc_body(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body)
        {
            int i = 0;
            bool ret = false;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            while (i < tarr.size())
            {
                int end = tarr.search(IsSameCriteria(TokenType::ENDL), i);
                if (end < 0)
                    end = tarr.size();
                else
                    ret = ret || parsebc_instruction(errs, { tarr, i, end }, mod, body);
                for (i = end; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            }
            return ret;
        }

        bool parsebc_module(Errors& errs, const char* fname, char* buf, size_t bufsz, ByteCodeModule& mod)
        {
            TokenArray tarr;
            if (lex_buf(errs, fname, buf, bufsz, tarr))
                return true;

            bool ret = false;
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            while (i < tarr.size())
            {
                if (tarr[i++]->expect<TokenType::DOT>(errs))
                    return true;
                if (tarr[i]->expect<TokenType::IDN>(errs))
                    return true;
                std::string name = tarr[i++]->get<TokenType::IDN>().val;
                if (tarr[i++]->expect<TokenType::ENDL>(errs))
                    return true;
                int end = tarr.search(IsSameCriteria(TokenType::DOT), i);
                if (end < 0)
                    end = tarr.size();
                ByteCodeBody body;
                if (parsebc_body(errs, { tarr, i, end }, mod, body))
                    ret = true;
                else
                    ret = ret || mod.add_block(errs, name, body);
                i = end;
            }
            return ret;
        }

        bool parsebc_module(Errors& errs, const char* fname, FILE* pf, ByteCodeModule& mod)
        {
            fseek(pf, 0, SEEK_END);
            size_t fsz = ftell(pf);
            rewind(pf);
            char* pbuf = new char[fsz + 1];
            size_t result = fread(pbuf, 1, fsz, pf);
            if (result != fsz)
            {
                delete[] pbuf;
                throw std::runtime_error("fread failed");
            }
            pbuf[fsz] = '\0';
            bool ret = parsebc_module(errs, fname, pbuf, fsz + 1, mod);
            delete[] pbuf;
            return ret;
        }
    }
}
