#include <ned/lang/bytecode.h>
#include <ned/lang/lexer.h>

#include <vector>
#include <string>
#include <tuple>
#include <iterator>
#include <cstdio>

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
        std::vector<WordInfo>::const_iterator parse_instruction(
            ParsingErrors& errs, StaticsBuilder& statics, CodeBlock& block,
            std::vector<WordInfo>::const_iterator& it,
            const std::vector<WordInfo>::const_iterator& end)
        {
            switch (hash(it->word))
            {
            case hash("jmp"):
                if (std::distance(it, end) < 2)  // Should be O(1) for vector
                    return errs.add(it->line, it->col, "Invalid 'jmp' instruction");
                block.add_instruction(instruction::Jmp((it + 1)->word));
                it += 2;
                return false;
            case hash("brt"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'brt' instruction");
                block.add_instruction(instruction::Brt((it + 1)->word));
                it += 2;
                return false;
            case hash("brf"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'brf' instruction");
                block.add_instruction(instruction::Brf((it + 1)->word));
                it += 2;
                return false;
            case hash("pop"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'brt' instruction");
                block.add_instruction(instruction::Pop(std::stoi((it - 1)->word)));
                it += 2;
                return false;
            case hash("new"):
                if (std::distance(it, end) < 3)
                    return errs.add(it->line, it->col, "Invalid 'new' instruction");
                block.add_instruction(instruction::New(statics.get((it + 1)->word, (it + 2)->word)));
                it += 3;
                return false;
            case hash("arr"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'arr' instruction");
                block.add_instruction(instruction::Arr(std::stoi((it + 1)->word)));
                it += 2;
                return false;
            case hash("tup"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'tup' instruction");
                block.add_instruction(instruction::Tup(std::stoi((it + 1)->word)));
                it += 2;
                return false;
            case hash("inst"):
                block.add_instruction(instruction::Inst());
                it += 1;
                return false;
            case hash("type"):
                block.add_instruction(instruction::Type());
                it += 1;
                return false;
            case hash("dup"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'dup' instruction");
                block.add_instruction(instruction::Dup(std::stoi((it + 1)->word)));
                it += 2;
                return false;
            case hash("carg"):
                if (std::distance(it, end) < 2)
                    return errs.add(it->line, it->col, "Invalid 'carg' instruction");
                block.add_instruction(instruction::Carg(std::stoi((it + 1)->word)));
                it += 2;
                return false;
            case hash("call"):
                block.add_instruction(instruction::Call());
                it += 1;
                return false;
            case hash("ret"):
                block.add_instruction(instruction::Ret());
                it += 1;
                return false;
            case hash("add"):
                block.add_instruction(instruction::Add());
                it += 1;
                return false;
            case hash("sub"):
                block.add_instruction(instruction::Sub());
                it += 1;
                return false;
            case hash("mul"):
                block.add_instruction(instruction::Mul());
                it += 1;
                return false;
            case hash("div"):
                block.add_instruction(instruction::Div());
                it += 1;
                return false;
            case hash("mod"):
                block.add_instruction(instruction::Mod());
                it += 1;
                return false;
            case hash("eq"):
                block.add_instruction(instruction::Eq());
                it += 1;
                return false;
            case hash("ne"):
                block.add_instruction(instruction::Ne());
                it += 1;
                return false;
            case hash("ge"):
                block.add_instruction(instruction::Ge());
                it += 1;
                return false;
            case hash("le"):
                block.add_instruction(instruction::Le());
                it += 1;
                return false;
            case hash("gt"):
                block.add_instruction(instruction::Gt());
                it += 1;
                return false;
            case hash("lt"):
                block.add_instruction(instruction::Lt());
                it += 1;
                return false;
            case hash("idx"):
                block.add_instruction(instruction::Idx());
                it += 1;
                return false;
            default:
                return errs.add(it->line, it->col, "Unrecognized opcode: {}", it->word);
            }
        }

        bool parsebc_block(ParsingErrors& errs, const TokenArray& tarr, StaticsBuilder& statics, CodeBlock& block)
        {
            auto words = read_words(start, end);
            for (auto it = words.begin(); it != words.end(); it++)
            {
                if (it->word[0] == ':')
                {
                    block.add_label(it->word.substr(1));
                    it++;
                }
                else
                    it = parse_instruction(block, statics, it, words.end());
            }
        }

        bool parsebc_module(ParsingErrors& errs, const char* fname, char* buf, size_t bufsz, StaticsBuilder& statics, Module& mod)
        {
            TokenArray tarr;
            if (lex_buf(errs, fname, buf, bufsz, tarr))
                return true;

            // Initialization for the 
            int i = 0;
            for (; i < tarr.size() && tarr[i]->is_whitespace(); i++);
            while (i < tarr.size())
            {
                if (tarr[i]->expect<TokenType::DOT>(errs))
                    return true;
                int end = tarr.search(IsSameCriteria(TokenType::COLON), i);
                if (end == -1)
                    return errs.add(tarr[i], "Missing ':' in code block signature");
                if (end <= i + 2)  // Needs at least a code block type (struct/fn/def) and a name
                    return errs.add(tarr[i], "Empty signatures are not allowed");
                
                if (parsebc_signature(errs, { tarr, i + 1, end }))
                    return true;

                i = end + 1;
                end = tarr.search(IsSameCriteria(TokenType::DOT), i);

            }
            return false;

            StaticsBuilder statics{};
            char* start = 0;
            char* end = buf;
            do
            {
                while (buf[start] && is_whitespace(buf[start])) start++;
                if (buf[start] != '.')
                    return errs.add(line_num, col_num, "Invalid start of bytecode block");
                end = start;
                while (buf[end] && buf[end] != ':') end++;
                if (buf[end] != ':')
                    return errs.add(line_num, col)
                TokenArray tarr;
                lex_buf(errs, fname, buf, end - buf, tarr, line_num, col_num);

                while (*end && *end != '.') end++;
                if (parsebc_block(errs, statics, start, end))
                    return true;
            } while (*end);
            return false;
        }

        bool read_bcfile(ParsingErrors& errs, const char* fname, FILE* pf)
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
            bool ret = parsebc_module(errs, fname, pbuf);
            delete[] pbuf;
            return ret;
        }
    }
}
