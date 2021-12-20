#include <ned/lang/bytecode.h>
#include <ned/lang/lexer.h>

#include <cstdio>
#include <vector>
#include <string>
#include <tuple>
#include <iterator>

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

size_t line_num = 1;
size_t col_num = 0;
inline bool is_whitespace(char c)
{
    col_num++;
    if (c == '\n')
    {
        line_num++;
        col_num = 1;
        return true;
    }
    return
        c == ' ' ||
        c == '\t' ||
        c == '\r' ||
        c == '\v' ||
        c == '\f';
}

char* read_word(char* buf, char* word)
{
    if (*buf == ';')
    {
        while (*buf && *buf != '\n') buf++;
        line_num++;
        col_num = 1;
    }
    if (!*buf)
    {
        *word = '\0';
        return buf;
    }
    char* pw = word;
    while (*buf && !is_whitespace(*buf))
    {
        *pw = *buf;
        pw++;
        buf++;
    }
    *pw = '\0';
    return buf;
}

struct WordInfo
{
    size_t line;
    size_t col;
    std::string word;
};

std::vector<WordInfo> read_words(char* start, char* end)
{
    while (start != end && is_whitespace(*start)) start++;
    std::vector<WordInfo> result;
    if (start >= end)
        return result;
    char word[128];
    while (1)
    {
        size_t line = line_num;
        size_t col = col_num;
        while (start < end && is_whitespace(*start)) start++;
        start = read_word(start, word);
        if (start >= end)
            return result;
        result.push_back({ line, col, word });
    }
}

namespace nn
{
    namespace lang
    {
        void* parse_signature(char* buf)
        {
            while (*buf && is_whitespace(*buf)) buf++;
            if (*buf != '.')
                throw SyntaxError(line_num, col_num, "Invalid start of bytecode block");
            
            if (!*++buf)
                throw SyntaxError(line_num, col_num, "Invalid start of bytecode block");
            char type_name[128];

            char* end = buf;
            TokenArray tarr;
            lex_buf("bytecode module", buf, end - buf, tarr, line_num, col_num);
            // TODO: parse block signatures
        }

        std::vector<WordInfo>::const_iterator parse_instruction(
            CodeBlock& block, StaticsBuilder& statics,
            std::vector<WordInfo>::const_iterator it,
            std::vector<WordInfo>::const_iterator end)
        {
            switch (hash(it->word))
            {
            case hash("jmp"):
                if (std::distance(it, end) < 2)  // Should be O(1) for vector
                    throw SyntaxError(it->line, it->col, "Invalid 'jmp' instruction");
                block.add_instruction(instruction::Jmp((it + 1)->word));
                return it + 2;
            case hash("brt"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'brt' instruction");
                block.add_instruction(instruction::Brt((it + 1)->word));
                return it + 2;
            case hash("brf"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'brf' instruction");
                block.add_instruction(instruction::Brf((it + 1)->word));
                return it + 2;
            case hash("pop"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'brt' instruction");
                block.add_instruction(instruction::Pop(std::stoi((it - 1)->word)));
                return it + 2;
            case hash("new"):
                if (std::distance(it, end) < 3)
                    throw SyntaxError(it->line, it->col, "Invalid 'new' instruction");
                block.add_instruction(instruction::New(statics.get((it + 1)->word, (it + 2)->word)));
                return it + 3;
            case hash("arr"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'arr' instruction");
                block.add_instruction(instruction::Arr(std::stoi((it + 1)->word)));
                return it + 2;
            case hash("tup"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'tup' instruction");
                block.add_instruction(instruction::Tup(std::stoi((it + 1)->word)));
                return it + 2;
            case hash("inst"):
                block.add_instruction(instruction::Inst());
                return it + 1;
            case hash("type"):
                block.add_instruction(instruction::Type());
                return it + 1;
            case hash("dup"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'dup' instruction");
                block.add_instruction(instruction::Dup(std::stoi((it + 1)->word)));
                return it + 2;
            case hash("carg"):
                if (std::distance(it, end) < 2)
                    throw SyntaxError(it->line, it->col, "Invalid 'carg' instruction");
                block.add_instruction(instruction::Carg(std::stoi((it + 1)->word)));
                return it + 2;
            case hash("call"):
                block.add_instruction(instruction::Call());
                return it + 1;
            case hash("ret"):
                block.add_instruction(instruction::Ret());
                return it + 1;
            case hash("add"):
                block.add_instruction(instruction::Add());
                return it + 1;
            case hash("sub"):
                block.add_instruction(instruction::Sub());
                return it + 1;
            case hash("mul"):
                block.add_instruction(instruction::Mul());
                return it + 1;
            case hash("div"):
                block.add_instruction(instruction::Div());
                return it + 1;
            case hash("mod"):
                block.add_instruction(instruction::Mod());
                return it + 1;
            case hash("eq"):
                block.add_instruction(instruction::Eq());
                return it + 1;
            case hash("ne"):
                block.add_instruction(instruction::Ne());
                return it + 1;
            case hash("ge"):
                block.add_instruction(instruction::Ge());
                return it + 1;
            case hash("le"):
                block.add_instruction(instruction::Le());
                return it + 1;
            case hash("gt"):
                block.add_instruction(instruction::Gt());
                return it + 1;
            case hash("lt"):
                block.add_instruction(instruction::Lt());
                return it + 1;
            case hash("idx"):
                block.add_instruction(instruction::Idx());
                return it + 1;
            default:
                throw SyntaxError(it->line, it->col, "Unrecognized opcode: {}", it->word);
            }
        }

        CodeBlock parse_block(StaticsBuilder& statics, char* start, char* end)
        {
            CodeBlock block{};
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

        void parse_module(char* buf)
        {
            StaticsBuilder statics{};
            char* start;
            char* end = buf;
            do
            {
                end = start = parse_signature(end);
                while (*end && *end != '.') end++;
                parse_block(statics, start, end);
            } while (*end);
        }

        void read_bcfile(FILE* pf)
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
            parse_module(pbuf);
            delete[] pbuf;
        }
    }
}
