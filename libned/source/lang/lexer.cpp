#include <ned/lang/lexer.h>

#include <stdexcept>
#include <cassert>
#include <cmath>

#include <iostream>

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

inline bool is_numeric(char c)
{
    return '0' <= c && c <= '9';
}

inline bool is_idnstart(char c)
{
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

inline bool is_idnchar(char c)
{
    return is_numeric(c) || is_idnstart(c);
}

namespace nn
{
    namespace lang
    {
        constexpr std::string to_string(TokenType ty) noexcept
        {
            switch (ty)
            {
            case TokenType::INVALID:
                return "INVALID TOKEN - LEXER BUG";
            case TokenType::INDENT:
                return "indent '\\t'";
            case TokenType::ENDL:
                return "end of line '\\n'";
            case TokenType::ANGLE_O:
                return "opened angle bracket '<'";
            case TokenType::ANGLE_C:
                return "closed angle bracket '>'";
            case TokenType::ROUND_O:
                return "opened round bracket '('";
            case TokenType::ROUND_C:
                return "closed round bracket ')'";
            case TokenType::SQUARE_O:
                return "opened square bracket '['";
            case TokenType::SQUARE_C:
                return "closed square bracket ']'";
            case TokenType::DOT:
                return "dot operator '.'";
            case TokenType::ELLIPSES:
                return "ellipses '...'";
            case TokenType::ARROW:
                return "arrow token '->'";
            case TokenType::COLON:
                return "colon character ':'";
            case TokenType::COMMA:
                return "comma character ','";
            case TokenType::MODE:
                return "context declaration '!'";
            case TokenType::ADD:
                return "addition operator '+'";
            case TokenType::SUB:
                return "substraction operator '-'";
            case TokenType::STAR:
                return "star operator '*'";
            case TokenType::DIV:
                return "division operator '/'";
            case TokenType::MOD:
                return "modulus operator '%'";
            case TokenType::POW:
                return "power operator '^'";
            case TokenType::IADD:
                return "assignment addition operator '+='";
            case TokenType::ISUB:
                return "assignment subtraction operator '-='";
            case TokenType::IMUL:
                return "assignment multiplication operator'*='";
            case TokenType::IDIV:
                return "assignment division operator '/='";
            case TokenType::IMOD:
                return "assignment modulus operator '%='";
            case TokenType::IPOW:
                return "assignment power operator '^='";
            case TokenType::ASSIGN:
                return "assignment operator '='";
            case TokenType::CMP_EQ:
                return "equality operator '=='";
            case TokenType::CMP_NE:
                return "inequality operator '!='";
            case TokenType::CMP_GE:
                return "greater than or equal operator '>='";
            case TokenType::CMP_LE:
                return "less than or equal operator '<='";
            case TokenType::LIT_INT:
                return "integer literal";
            case TokenType::LIT_FLOAT:
                return "float literal";
            case TokenType::LIT_STR:
                return "string literal";
            case TokenType::IDN:
                return "identifier";
            case TokenType::KW_NAMESPACE:
                return "keyword namespace";
            case TokenType::KW_STRUCT:
                return "keyword struct";
            case TokenType::KW_ENUM:
                return "keyword enum";
            case TokenType::KW_DEF:
                return "keyword def";
            case TokenType::KW_INTR:
                return "keyword intr";
            case TokenType::KW_FN:
                return "keyword fn";
            case TokenType::KW_INIT:
                return "keyword init";
            case TokenType::KW_RETURN:
                return "keyword return";
            case TokenType::KW_IMPORT:
                return "keyword import";
            case TokenType::KW_WHILE:
                return "keyword while";
            case TokenType::KW_FOR:
                return "keyword for";
            case TokenType::KW_IN:
                return "keyword in";
            case TokenType::KW_BREAK:
                return "keyword break";
            case TokenType::KW_CONTINUE:
                return "keyword continue";
            case TokenType::KW_MATCH:
                return "keyword match";
            case TokenType::KW_IF:
                return "keyword if";
            case TokenType::KW_ELIF:
                return "keyword elif";
            case TokenType::KW_ELSE:
                return "keyword else";
            case TokenType::KW_TYPE:
                return "keyword type";
            case TokenType::KW_FTY:
                return "keyword fty";
            case TokenType::KW_BOOL:
                return "keyword bool";
            case TokenType::KW_INT:
                return "keyword int";
            case TokenType::KW_FLOAT:
                return "keyword float";
            case TokenType::KW_STR:
                return "keyword str";
            case TokenType::KW_ARRAY:
                return "keyword array";
            case TokenType::KW_TUPLE:
                return "keyword tuple";
            case TokenType::KW_CFG:
                return "keyword cfg";
            case TokenType::KW_REF:
                return "keyword ref";
            case TokenType::KW_CONST:
                return "keyword const";
            case TokenType::KW_TRUE:
                return "keyword true";
            case TokenType::KW_FALSE:
                return "keyword false";
            case TokenType::KW_EXPORT:
                return "keyword export";
            case TokenType::KW_EXTERN:
                return "keyword extern";
            case TokenType::KW_FORWARD:
                return "keyword forward";
            case TokenType::KW_BACKWARD:
                return "keyword backward";
            case TokenType::KW_F16:
                return "keyword f16";
            case TokenType::KW_F32:
                return "keyword f32";
            case TokenType::KW_F64:
                return "keyword f64";
            case TokenType::KW_AND:
                return "keyword and";
            case TokenType::KW_OR:
                return "keyword or";
            case TokenType::KW_NOT:
                return "keyword not";
            case TokenType::KW_ADD_INTR_INFO:
                return "keyword __add_intr_info";
            default:
                return "UNKNOWN TOKEN - LEXER BUG";
            }
        }

        std::string to_string(const Token* ptk) noexcept
        {
            switch (ptk->ty)
            {
            case TokenType::INVALID:
                return "\nINVALID\n";
            case TokenType::INDENT:
                return "    ";
            case TokenType::ENDL:
                return "\n";
            case TokenType::ANGLE_O:
                return "<";
            case TokenType::ANGLE_C:
                return ">";
            case TokenType::ROUND_O:
                return "(";
            case TokenType::ROUND_C:
                return ")";
            case TokenType::SQUARE_O:
                return "[";
            case TokenType::SQUARE_C:
                return "]";
            case TokenType::DOT:
                return ".";
            case TokenType::ELLIPSES:
                return "...";
            case TokenType::ARROW:
                return " -> ";
            case TokenType::COLON:
                return ":";
            case TokenType::COMMA:
                return ", ";
            case TokenType::MODE:
                return "!";
            case TokenType::ADD:
                return " + ";
            case TokenType::SUB:
                return " - ";
            case TokenType::STAR:
                return "*";
            case TokenType::DIV:
                return " / ";
            case TokenType::MOD:
                return " % ";
            case TokenType::POW:
                return " ^ ";
            case TokenType::IADD:
                return " += ";
            case TokenType::ISUB:
                return " -= ";
            case TokenType::IMUL:
                return " *= ";
            case TokenType::IDIV:
                return " /= ";
            case TokenType::IMOD:
                return " %= ";
            case TokenType::IPOW:
                return " ^= ";
            case TokenType::ASSIGN:
                return " = ";
            case TokenType::CMP_EQ:
                return " == ";
            case TokenType::CMP_NE:
                return " != ";
            case TokenType::CMP_GE:
                return " >= ";
            case TokenType::CMP_LE:
                return " <= ";
            case TokenType::LIT_INT:
                return std::to_string(static_cast<const TokenImp<TokenType::LIT_INT>*>(ptk)->val);
            case TokenType::LIT_FLOAT:
                return std::to_string(static_cast<const TokenImp<TokenType::LIT_FLOAT>*>(ptk)->val);
            case TokenType::LIT_STR:
                return std::string("\"") + static_cast<const TokenImp<TokenType::LIT_STR>*>(ptk)->val + "\"";
            case TokenType::IDN:
                return std::string(static_cast<const TokenImp<TokenType::IDN>*>(ptk)->val);
            case TokenType::KW_NAMESPACE:
                return "namespace ";
            case TokenType::KW_STRUCT:
                return "struct ";
            case TokenType::KW_ENUM:
                return "enum ";
            case TokenType::KW_DEF:
                return "def ";
            case TokenType::KW_INTR:
                return "intr ";
            case TokenType::KW_FN:
                return "fn ";
            case TokenType::KW_INIT:
                return "init ";
            case TokenType::KW_RETURN:
                return "return ";
            case TokenType::KW_IMPORT:
                return "import ";
            case TokenType::KW_WHILE:
                return "while ";
            case TokenType::KW_FOR:
                return "for ";
            case TokenType::KW_IN:
                return "in ";
            case TokenType::KW_BREAK:
                return "break ";
            case TokenType::KW_CONTINUE:
                return "continue ";
            case TokenType::KW_MATCH:
                return "match ";
            case TokenType::KW_IF:
                return "if ";
            case TokenType::KW_ELIF:
                return "elif ";
            case TokenType::KW_ELSE:
                return "else ";
            case TokenType::KW_TYPE:
                return "type ";
            case TokenType::KW_FTY:
                return "fty ";
            case TokenType::KW_BOOL:
                return "bool ";
            case TokenType::KW_INT:
                return "int ";
            case TokenType::KW_FLOAT:
                return "float ";
            case TokenType::KW_STR:
                return "str ";
            case TokenType::KW_ARRAY:
                return "array ";
            case TokenType::KW_TUPLE:
                return "tuple ";
            case TokenType::KW_CFG:
                return "cfg ";
            case TokenType::KW_REF:
                return "ref ";
            case TokenType::KW_CONST:
                return "const ";
            case TokenType::KW_TRUE:
                return "true ";
            case TokenType::KW_FALSE:
                return "false ";
            case TokenType::KW_EXPORT:
                return "export ";
            case TokenType::KW_EXTERN:
                return "extern ";
            case TokenType::KW_FORWARD:
                return "forward";
            case TokenType::KW_BACKWARD:
                return "backward";
            case TokenType::KW_F16:
                return "f16 ";
            case TokenType::KW_F32:
                return "f32 ";
            case TokenType::KW_F64:
                return "f64 ";
            case TokenType::KW_AND:
                return " and ";
            case TokenType::KW_OR:
                return " or ";
            case TokenType::KW_NOT:
                return " not ";
            case TokenType::KW_ADD_INTR_INFO:
                return "__add_intr_info";
            default:
                return "unknown";
            }
        }

        TokenArray::TokenArray(size_t mem_sz, size_t off_cap)
        {
            this->is_slice = false;

            this->mem_sz = mem_sz;
            this->rawlen = 0;
            this->pbuf = (uint8_t*)std::malloc(mem_sz);
            if (!this->pbuf)
                throw std::bad_alloc();

            this->off_cap = off_cap;
            this->off_len = 0;
            this->offsets = (size_t*)std::malloc(sizeof(size_t) * off_cap);
            if (!this->offsets)
                throw std::bad_alloc();
        }

        TokenArray::TokenArray(const TokenArray& base, int start)
        {
            size_t base_offset = base.offsets[start];
            size_t base_buflen = ((Token*)(base.pbuf + base.offsets[base.size() - 1]))->sz + base.offsets[base.size() - 1];

            this->is_slice = true;

            int end = (int)base.size();
            this->mem_sz = 0;
            this->rawlen = base_buflen - base_offset;
            this->pbuf = base.pbuf + base_offset;

            this->off_cap = 0;
            this->off_len = base.size() - start;
            this->offsets = (size_t*)std::malloc(sizeof(size_t) * this->off_len);
            if (!this->offsets)
                throw std::bad_alloc();

            for (int i = 0; i < this->off_len; i++)
                this->offsets[i] = base.offsets[start + i] - base_offset;
        }

        TokenArray::TokenArray(const TokenArray& base, int start, int end)
        {
            if (end < 0)
                end += (int)base.size();

            size_t base_offset = base.offsets[start];
            
            this->is_slice = true;

            this->mem_sz = 0;
            this->rawlen = base.offsets[end] - base_offset;
            this->pbuf = base.pbuf + base_offset;

            this->off_cap = 0;
            this->off_len = (size_t)end - start;
            this->offsets = (size_t*)std::malloc(sizeof(size_t) * this->off_len);
            if (!this->offsets)
                throw std::bad_alloc();

            for (int i = 0; i < this->off_len; i++)
                this->offsets[i] = base.offsets[(size_t)start + i] - base_offset;
        }

        TokenArray::~TokenArray()
        {
            // TODO: call release on each of the individual tokens
            if (!is_slice)
                std::free(pbuf);

            // offsets are always dynamically allocated
            std::free(offsets);
        }

        TokenArray::TokenArray(TokenArray&& tarr) noexcept
        {
            is_slice = tarr.is_slice;
            mem_sz = tarr.mem_sz;
            pbuf = tarr.pbuf;
            offsets = std::move(tarr.offsets);
            tarr.pbuf = nullptr;
        }
        
        TokenArray& TokenArray::operator=(TokenArray&& tarr) noexcept
        {
            if (this == &tarr)
                return *this;

            std::free(pbuf);

            is_slice = tarr.is_slice;
            mem_sz = tarr.mem_sz;
            pbuf = tarr.pbuf;
            offsets = std::move(tarr.offsets);

            tarr.pbuf = nullptr;
            return *this;
        }

        TokenArrayIterator TokenArray::begin()
        {
            return TokenArrayIterator((Token*)pbuf);
        }

        TokenArrayIterator TokenArray::end()
        {
            if (size() == 0)
                return reinterpret_cast<Token*>(pbuf);
            uint8_t* last = pbuf + offsets[off_len - 1];
            return TokenArrayIterator(reinterpret_cast<Token*>(last + reinterpret_cast<Token*>(last)->sz));
        }

        TokenArrayConstIterator TokenArray::begin() const
        {
            return TokenArrayConstIterator((Token*)pbuf);
        }

        TokenArrayConstIterator TokenArray::end() const
        {
            if (size() == 0)
                return reinterpret_cast<Token*>(pbuf);
            uint8_t* last = pbuf + offsets[off_len - 1];
            return TokenArrayConstIterator(reinterpret_cast<Token*>(last + reinterpret_cast<Token*>(last)->sz));
        }

        const Token* TokenArray::operator[](size_t idx) const noexcept
        {
            return (Token*)(pbuf + offsets[idx]);
        }

        size_t TokenArray::size() const
        {
            return off_len;
        }

        std::string TokenArray::to_string() const noexcept
        {
            std::string result;
            for (int i = 0; i < size(); i++)
                result += ::nn::lang::to_string((*this)[i]);
            result += "\n";
            return result;
        }

#ifdef _DEBUG
        void TokenArray::print() const
        {
            for (int i = 0; i < size(); i++)
                std::cout << ::nn::lang::to_string((*this)[i]);
            std::cout << std::endl;
        }
#endif

        bool lex_buf(const char* fname, char* buf, size_t bufsz, TokenArray& tarr, uint32_t line_num, int32_t line_start)
        {
            bool use_indents = true;

            for (uint32_t i = 0; i < bufsz;)
            {
                switch (buf[i])
                {
                case '#':
                    do { i++; } while (i < bufsz && buf[i] != '\n');
                    use_indents = true;
                    line_start = i;
                    line_num++;
                    tarr.push_back(TokenImp<TokenType::ENDL>(fname, line_num, i - line_start));
                    break;
                case ' ':
                    if (use_indents
                        && bufsz - i >= 4
                        && buf[i + 1] == ' '
                        && buf[i + 2] == ' '
                        && buf[i + 3] == ' '
                        )
                    {
                        tarr.push_back(TokenImp<TokenType::INDENT>(fname, line_num, i - line_start));
                        i += 4;
                        continue;
                    }
                    break;
                case '\t':
                    if (use_indents)
                        tarr.push_back(TokenImp<TokenType::INDENT>(fname, line_num, i - line_start));
                    line_start -= 4 - (i - line_start) % 4;  // Its a bit hacky, but it works
                    break;
                case '\r':
                case '\v':
                case '\f':
                    break;
                case '\n':
                    use_indents = true;
                    line_start = i;
                    line_num++;
                    tarr.push_back(TokenImp<TokenType::ENDL>(fname, line_num, i - line_start));
                    break;
                case '<':
                    use_indents = false;
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_LE>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    // this is so scuffed
                    if (i > 0 && buf[i - 1] == ' ')
                        tarr.push_back(TokenImp<TokenType::CMP_LT>(fname, line_num, i - line_start));
                    else
                        tarr.push_back(TokenImp<TokenType::ANGLE_O>(fname, line_num, i - line_start));
                    break;
                case '>':
                    use_indents = false;
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_GE>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    // I don't know how else to do it though
                    if (i > 0 && buf[i - 1] == ' ')
                        tarr.push_back(TokenImp<TokenType::CMP_GT>(fname, line_num, i - line_start));
                    else
                        tarr.push_back(TokenImp<TokenType::ANGLE_C>(fname, line_num, i - line_start));
                    break;
                case '(':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::ROUND_O>(fname, line_num, i - line_start));
                    break;
                case ')':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::ROUND_C>(fname, line_num, i - line_start));
                    break;
                case '[':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::SQUARE_O>(fname, line_num, i - line_start));
                    break;
                case ']':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::SQUARE_C>(fname, line_num, i - line_start));
                    break;
                case '.':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::DOT>(fname, line_num, i - line_start));
                    break;
                    if (bufsz - i >= 3 && buf[i + 1] == '.' && buf[i + 2] == '.')
                    {
                        tarr.push_back(TokenImp<TokenType::ELLIPSES>(fname, line_num, i - line_start));
                        i += 3;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ADD>(fname, line_num, i - line_start));
                    break;
                case ':':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == ':')
                    {
                        tarr.push_back(TokenImp<TokenType::CAST>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::COLON>(fname, line_num, i - line_start));
                    break;
                case ',':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::COMMA>(fname, line_num, i - line_start));
                    break;
                case '+':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IADD>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ADD>(fname, line_num, i - line_start));
                    break;
                case '*':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IMUL>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::STAR>(fname, line_num, i - line_start));
                    break;
                case '/':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IDIV>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::DIV>(fname, line_num, i - line_start));
                    break;
                case '%':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IMOD>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::MOD>(fname, line_num, i - line_start));
                    break;
                case '^':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IPOW>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::POW>(fname, line_num, i - line_start));
                    break;
                case '!':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_NE>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::MODE>(fname, line_num, i - line_start));
                    break;
                case '=':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_EQ>(fname, line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ASSIGN>(fname, line_num, i - line_start));
                    break;
                case '"':
                {
                    use_indents = false;

                    TokenImp<TokenType::LIT_STR> tk(fname, line_num, i - line_start);
                    int sidx = 0;
                    for (i += 1; i < bufsz && sidx < 256 && buf[i] != '"'; i++, sidx++)
                        tk.val[sidx] = buf[i];
                    if (i >= bufsz)
                        return error::syntax(tk, "Missing closing '\"' for string literal");
                    if (sidx == 256)
                        throw std::overflow_error("buffer overflow for string literal during lexing");
                    assert(buf[i] == '"');
                    tk.val[sidx] = '\0';
                    tarr.push_back(tk);
                    break;
                }
                default:
                {
                    use_indents = false;

                    // Handling numeric types
                    uint32_t col_num = i - line_start;
                    bool neg_val = false;
                    bool use_float = false;
                    if (buf[i] == '-')
                    {
                        if (bufsz - i >= 2)
                        {
                            if (buf[i + 1] == '=')
                            {
                                tarr.push_back(TokenImp<TokenType::ISUB>(fname, line_num, col_num));
                                i += 2;
                                continue;
                            }
                            else if (buf[i + 1] == '>')
                            {
                                tarr.push_back(TokenImp<TokenType::ARROW>(fname, line_num, col_num));
                                i += 2;
                                continue;
                            }
                        }
                        i += 1;
                        if (bufsz == i || (!is_numeric(buf[i]) && buf[i] != '.'))  // not a number
                        {
                            tarr.push_back(TokenImp<TokenType::SUB>(fname, line_num, col_num));
                            break;
                        }
                        neg_val = true;
                        if (buf[i] == '.')
                        {
                            use_float = true;
                            i += 1;
                        }
                    }
                    if (i >= bufsz)
                        return error::syntax(fname, line_num, col_num, "Unexpected EOF while lexing integral type");
                    if (is_numeric(buf[i]))
                    {
                        int64_t ival = 0;
                        if (!use_float)
                        {
                            while (i < bufsz && is_numeric(buf[i]))
                            {
                                ival *= 10;
                                ival += buf[i] - '0';
                                i += 1;
                            }
                            if (i < bufsz && buf[i] == '.')
                            {
                                use_float = true;
                                i += 1;
                            }
                            else if (i < bufsz && (buf[i] == 'e' || buf[i] == 'E'))
                            {
                                use_float = true;
                            }
                            else
                            {
                                if (neg_val)
                                    ival = -ival;
                                TokenImp<TokenType::LIT_INT> tk(fname, line_num, col_num);
                                tk.val = ival;
                                tarr.push_back(tk);
                                continue;
                            }
                        }

                        // use ival as the >1 portion of the float, and find the <1 portion
                        float multiplier = 0.1f;
                        double fval = (double)ival;
                        while (i < bufsz && is_numeric(buf[i]))
                        {
                            fval += multiplier * (buf[i] - '0');
                            multiplier /= 10;
                            i += 1;
                        }

                        if (i < bufsz && (buf[i] == 'e' || buf[i] == 'E'))
                        {
                            i += 1;
                            // lex the float point exponent as a signed int
                            bool negexp = false;
                            if (buf[i] == '-')
                            {
                                negexp = true;
                                i += 1;
                            }
                            int exp = 0;
                            while (is_numeric(buf[i]))
                            {
                                exp *= 10;
                                exp += buf[i] - '0';
                                i += 1;
                            }
                            if (negexp)
                                exp = -exp;
                            fval = std::pow(fval, (double)exp);
                        }

                        if (neg_val)
                            fval = -fval;
                        TokenImp<TokenType::LIT_FLOAT> tk(fname, line_num, col_num);
                        tk.val = fval;
                        tarr.push_back(tk);
                        continue;
                    }

                    // Only indentifiers and keywords are left
                    if (!is_idnstart(buf[i]))
                        return error::syntax(fname, line_num, col_num, "Unexpected character '%'", buf[i]);
                    
                    char idn_buf[64];
                    int iidx = 0;
                    for (; i < bufsz && iidx < 64 && is_idnchar(buf[i]); i++, iidx++)
                        idn_buf[iidx] = buf[i];
                    if (iidx == 64)
                        throw std::overflow_error("buffer overflow for identifier during lexing");
                    idn_buf[iidx] = '\0';

                    // checking for keywords
                    switch (hash(idn_buf))
                    {
                    case hash("namespace"):
                        tarr.push_back(TokenImp<TokenType::KW_NAMESPACE>(fname, line_num, col_num));
                        continue;
                    case hash("struct"):
                        tarr.push_back(TokenImp<TokenType::KW_STRUCT>(fname, line_num, col_num));
                        continue;
                    case hash("enum"):
                        tarr.push_back(TokenImp<TokenType::KW_ENUM>(fname, line_num, col_num));
                        continue;
                    case hash("def"):
                        tarr.push_back(TokenImp<TokenType::KW_DEF>(fname, line_num, col_num));
                        continue;
                    case hash("intr"):
                        tarr.push_back(TokenImp<TokenType::KW_INTR>(fname, line_num, col_num));
                        continue;
                    case hash("fn"):
                        tarr.push_back(TokenImp<TokenType::KW_FN>(fname, line_num, col_num));
                        continue;
                    case hash("init"):
                        tarr.push_back(TokenImp<TokenType::KW_INIT>(fname, line_num, col_num));
                        continue;
                    case hash("return"):
                        tarr.push_back(TokenImp<TokenType::KW_RETURN>(fname, line_num, col_num));
                        continue;
                    case hash("import"):
                        tarr.push_back(TokenImp<TokenType::KW_IMPORT>(fname, line_num, col_num));
                        continue;
                    case hash("while"):
                        tarr.push_back(TokenImp<TokenType::KW_WHILE>(fname, line_num, col_num));
                        continue;
                    case hash("for"):
                        tarr.push_back(TokenImp<TokenType::KW_FOR>(fname, line_num, col_num));
                        continue;
                    case hash("in"):
                        tarr.push_back(TokenImp<TokenType::KW_IN>(fname, line_num, col_num));
                        continue;
                    case hash("break"):
                        tarr.push_back(TokenImp<TokenType::KW_BREAK>(fname, line_num, col_num));
                        continue;
                    case hash("continue"):
                        tarr.push_back(TokenImp<TokenType::KW_CONTINUE>(fname, line_num, col_num));
                        continue;
                    case hash("match"):
                        tarr.push_back(TokenImp<TokenType::KW_MATCH>(fname, line_num, col_num));
                        continue;
                    case hash("if"):
                        tarr.push_back(TokenImp<TokenType::KW_IF>(fname, line_num, col_num));
                        continue;
                    case hash("elif"):
                        tarr.push_back(TokenImp<TokenType::KW_ELIF>(fname, line_num, col_num));
                        continue;
                    case hash("else"):
                        tarr.push_back(TokenImp<TokenType::KW_ELSE>(fname, line_num, col_num));
                        continue;
                    case hash("type"):
                        tarr.push_back(TokenImp<TokenType::KW_TYPE>(fname, line_num, col_num));
                        continue;
                    case hash("void"):
                        tarr.push_back(TokenImp<TokenType::KW_VOID>(fname, line_num, col_num));
                        continue;
                    case hash("fty"):
                        tarr.push_back(TokenImp<TokenType::KW_FTY>(fname, line_num, col_num));
                        continue;
                    case hash("bool"):
                        tarr.push_back(TokenImp<TokenType::KW_BOOL>(fname, line_num, col_num));
                        continue;
                    case hash("int"):
                        tarr.push_back(TokenImp<TokenType::KW_INT>(fname, line_num, col_num));
                        continue;
                    case hash("float"):
                        tarr.push_back(TokenImp<TokenType::KW_FLOAT>(fname, line_num, col_num));
                        continue;
                    case hash("str"):
                        tarr.push_back(TokenImp<TokenType::KW_STR>(fname, line_num, col_num));
                        continue;
                    case hash("array"):
                        tarr.push_back(TokenImp<TokenType::KW_ARRAY>(fname, line_num, col_num));
                        continue;
                    case hash("tuple"):
                        tarr.push_back(TokenImp<TokenType::KW_TUPLE>(fname, line_num, col_num));
                        continue;
                    case hash("cfg"):
                        tarr.push_back(TokenImp<TokenType::KW_CFG>(fname, line_num, col_num));
                        continue;
                    case hash("ref"):
                        tarr.push_back(TokenImp<TokenType::KW_REF>(fname, line_num, col_num));
                        continue;
                    case hash("const"):
                        tarr.push_back(TokenImp<TokenType::KW_CONST>(fname, line_num, col_num));
                        continue;
                    case hash("null"):
                        tarr.push_back(TokenImp<TokenType::KW_NULL>(fname, line_num, col_num));
                        continue;
                    case hash("true"):
                        tarr.push_back(TokenImp<TokenType::KW_TRUE>(fname, line_num, col_num));
                        continue;
                    case hash("false"):
                        tarr.push_back(TokenImp<TokenType::KW_FALSE>(fname, line_num, col_num));
                        continue;
                    case hash("export"):
                        tarr.push_back(TokenImp<TokenType::KW_EXPORT>(fname, line_num, col_num));
                        continue;
                    case hash("extern"):
                        tarr.push_back(TokenImp<TokenType::KW_EXTERN>(fname, line_num, col_num));
                        continue;
                    case hash("forward"):
                        tarr.push_back(TokenImp<TokenType::KW_FORWARD>(fname, line_num, col_num));
                        continue;
                    case hash("backward"):
                        tarr.push_back(TokenImp<TokenType::KW_BACKWARD>(fname, line_num, col_num));
                        continue;
                    case hash("f16"):
                        tarr.push_back(TokenImp<TokenType::KW_F16>(fname, line_num, col_num));
                        continue;
                    case hash("f32"):
                        tarr.push_back(TokenImp<TokenType::KW_F32>(fname, line_num, col_num));
                        continue;
                    case hash("f64"):
                        tarr.push_back(TokenImp<TokenType::KW_F64>(fname, line_num, col_num));
                        continue;
                    case hash("and"):
                        tarr.push_back(TokenImp<TokenType::KW_AND>(fname, line_num, col_num));
                        continue;
                    case hash("or"):
                        tarr.push_back(TokenImp<TokenType::KW_OR>(fname, line_num, col_num));
                        continue;
                    case hash("not"):
                        tarr.push_back(TokenImp<TokenType::KW_NOT>(fname, line_num, col_num));
                        continue;
                    case hash("__add_intr_info"):
                        tarr.push_back(TokenImp<TokenType::KW_ADD_INTR_INFO>(fname, line_num, col_num));
                        continue;
                    }
                    
                    TokenImp<TokenType::IDN> tk(fname, line_num, col_num);
                    strcpy(tk.val, idn_buf);
                    tarr.push_back(tk);
                    continue;
                }
                }

                i++;
            }
            
            return false;
        }

        bool lex_file(const char* fname, TokenArray& tarr)
        {
            // temp, bad implmentation
            FILE* pf = fopen(fname, "rb");
            if (!pf)
                return error::general("Unable to open file '%'", fname);
            fseek(pf, 0, SEEK_END);
            size_t fsz = ftell(pf);
            rewind(pf);
            char* pbuf = new char[fsz + 1];
            if (!pbuf)
            {
                fclose(pf);
                return error::general("Out of memory");
            }
            size_t result = fread(pbuf, 1, fsz, pf);
            fclose(pf);
            if (result != fsz)
            {
                delete[] pbuf;
                error::general("Unable to read file '%'", fname);
                return true;
            }
            pbuf[fsz] = '\0';
            bool ret = lex_buf(fname, pbuf, fsz, tarr);
            delete[] pbuf;
            return ret;
        }

        /*
         *   Search criteria
         */

        void BracketCounter::count_token(const Token* ptk)
        {
            switch (ptk->ty)
            {
            case TokenType::ROUND_O:
                rbrac++;
                break;
            case TokenType::ROUND_C:
                rbrac--;
                break;
            case TokenType::SQUARE_O:
                sbrac++;
                break;
            case TokenType::SQUARE_C:
                sbrac--;
                break;
            case TokenType::ANGLE_O:
                abrac++;
                break;
            case TokenType::ANGLE_C:
                abrac--;
                break;
            }
        }

        bool BracketCounter::in_bracket() const
        {
            return rbrac != 0 || sbrac != 0 || abrac != 0;
        }

        IsSameCriteria::IsSameCriteria(TokenType ty) : ty(ty) {}

        int IsSameCriteria::accept(const Token* ptk, int idx)
        {
            count_token(ptk);
            return (idx + 1) * (!in_bracket() && ptk->ty == ty) - 1;
        }

        IsInCriteria::IsInCriteria(const std::vector<TokenType>& tys)
        {
            this->tys = tys;
        }

        int IsInCriteria::accept(const Token* ptk, int idx)
        {
            count_token(ptk);
            return (idx + 1) * (!in_bracket() && std::find(tys.begin(), tys.end(), ptk->ty) != tys.end()) - 1;
        }

        ArgEndCriteria::ArgEndCriteria(TokenType close) : close(close) {}

        int ArgEndCriteria::accept(const Token* ptk, int idx)
        {
            if (!in_bracket() && (ptk->ty == close || ptk->ty == TokenType::COMMA))
                return idx;
            count_token(ptk);
            return -1;
        }

        int BlockStartCriteria::accept(const Token* ptk, int idx)
        {
            if (ptk->ty == TokenType::COLON)
            {
                if (!in_bracket())
                    return idx;
            }
            else
                count_token(ptk);
            return -1;
        }

        LineEndCriteria::LineEndCriteria(int indent_level) : target_ilv(indent_level) {}

        int LineEndCriteria::accept(const Token* ptk, int idx)
        {
            if (ptk->ty == TokenType::ENDL)
            {
                if (!in_bracket())
                {
                    last_endl = idx;
                    current_ilv = 0;
                    in_line = false;
                }
            }
            else if (ptk->ty == TokenType::INDENT)
                current_ilv++;
            else
            {
                if (!in_bracket() && !in_line && last_endl > 0 && current_ilv <= target_ilv)  // Must be >, not >=
                    return last_endl;
                in_line = true;
                count_token(ptk);
            }
            return -1;
        }
    }
}
