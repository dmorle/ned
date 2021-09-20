#include <ned/lang/lexer.h>

#include <stdexcept>
#include <cassert>
#include <cmath>

#ifdef _DEBUG
#include <iostream>
#endif

// macros for manual code generation
#define SMALL_PUSH_BACK(TY)                                                    \
template<>                                                                     \
void TokenArray::push_back(const TokenImp<TokenType::TY>& tk){                 \
    constexpr size_t nsz = sizeof(TokenImp<TokenType::TY>);                    \
    if (is_slice) throw std::runtime_error("TokenArray slices are immutable"); \
    void* tmp;                                                                 \
    if (rawlen + nsz >= mem_sz){                                               \
        mem_sz += mem_sz;                                                      \
        tmp = std::realloc(pbuf, mem_sz);                                      \
        if (!tmp) throw std::bad_alloc();                                      \
        pbuf = (uint8_t*)tmp;                                                  \
    }                                                                          \
    off_len++;                                                                 \
    if (off_len == off_cap){                                                   \
        off_cap += off_cap;                                                    \
        tmp = std::realloc(offsets, sizeof(size_t) * off_cap);                 \
        if (!tmp) throw std::bad_alloc();                                      \
        offsets = (size_t*)tmp;                                                \
    }                                                                          \
    offsets[off_len - 1] = rawlen;                                             \
    std::memcpy(pbuf + rawlen, &tk, nsz);                                      \
    rawlen += nsz;                                                             \
}

#define LARGE_PUSH_BACK(TY)                                                    \
template<>                                                                     \
void TokenArray::push_back(const TokenImp<TokenType::TY>& tk){                 \
    constexpr size_t nsz = sizeof(TokenImp<TokenType::TY>);                    \
    if (is_slice) throw std::runtime_error("TokenArray slices are immutable"); \
    void* tmp;                                                                 \
    while (rawlen + nsz >= mem_sz){                                            \
        mem_sz += mem_sz;                                                      \
        tmp = std::realloc(pbuf, mem_sz);                                      \
        if (!tmp) throw std::bad_alloc();                                      \
        pbuf = (uint8_t*)tmp;                                                  \
    }                                                                          \
    off_len++;                                                                 \
    if (off_len == off_cap){                                                   \
        off_cap += off_cap;                                                    \
        tmp = std::realloc(offsets, sizeof(size_t) * off_cap);                 \
        if (!tmp) throw std::bad_alloc();                                      \
        offsets = (size_t*)tmp;                                                \
    }                                                                          \
    offsets[off_len - 1] = rawlen;                                             \
    std::memcpy(pbuf + rawlen, &tk, nsz);                                      \
    rawlen += nsz;                                                             \
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
        std::string to_string(const Token* ptk)
        {
            switch (ptk->ty)
            {
            case TokenType::INVALID:
                return "invalid";
            case TokenType::INDENT:
                return "tab";
            case TokenType::ENDL:
                return "newline";
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
            case TokenType::COLON:
                return ":";
            case TokenType::COMMA:
                return ",";
            case TokenType::ADD:
                return "+";
            case TokenType::SUB:
                return "-";
            case TokenType::STAR:
                return "*";
            case TokenType::DIV:
                return "/";
            case TokenType::IADD:
                return "+=";
            case TokenType::ISUB:
                return "-=";
            case TokenType::IMUL:
                return "*=";
            case TokenType::IDIV:
                return "/=";
            case TokenType::ASSIGN:
                return "=";
            case TokenType::CMP_EQ:
                return "==";
            case TokenType::CMP_NE:
                return "!=";
            case TokenType::CMP_GE:
                return ">=";
            case TokenType::CMP_LE:
                return "<=";
            case TokenType::INT:
                return "int";
            case TokenType::FLOAT:
                return "float";
            case TokenType::STRLIT:
                return "string";
            case TokenType::IDN:
                return "identifier";
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

            int end = base.size();
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
                end += base.size();

            size_t base_offset = base.offsets[start];
            
            this->is_slice = true;

            this->mem_sz = 0;
            this->rawlen = base.offsets[end] - base_offset;
            this->pbuf = base.pbuf + base_offset;

            this->off_cap = 0;
            this->off_len = end - start;
            this->offsets = (size_t*)std::malloc(sizeof(size_t) * this->off_len);
            if (!this->offsets)
                throw std::bad_alloc();

            for (int i = 0; i < this->off_len; i++)
                this->offsets[i] = base.offsets[start + i] - base_offset;
        }

        TokenArray::~TokenArray()
        {
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

        const Token* TokenArray::operator[](size_t idx) const noexcept
        {
            return (Token*)(pbuf + offsets[idx]);
        }

        SMALL_PUSH_BACK( INDENT   );
        SMALL_PUSH_BACK( ENDL     );
        SMALL_PUSH_BACK( ANGLE_O  );
        SMALL_PUSH_BACK( ANGLE_C  );
        SMALL_PUSH_BACK( ROUND_O  );
        SMALL_PUSH_BACK( ROUND_C  );
        SMALL_PUSH_BACK( SQUARE_O );
        SMALL_PUSH_BACK( SQUARE_C );
        SMALL_PUSH_BACK( DOT      );
        SMALL_PUSH_BACK( COLON    );
        SMALL_PUSH_BACK( COMMA    );
        SMALL_PUSH_BACK( ADD      );
        SMALL_PUSH_BACK( SUB      );
        SMALL_PUSH_BACK( STAR     );
        SMALL_PUSH_BACK( DIV      );
        SMALL_PUSH_BACK( IADD     );
        SMALL_PUSH_BACK( ISUB     );
        SMALL_PUSH_BACK( IMUL     );
        SMALL_PUSH_BACK( IDIV     );
        SMALL_PUSH_BACK( ASSIGN   );
        SMALL_PUSH_BACK( CMP_EQ   );
        SMALL_PUSH_BACK( CMP_NE   );
        SMALL_PUSH_BACK( CMP_GE   );
        SMALL_PUSH_BACK( CMP_LE   );
        SMALL_PUSH_BACK( INT      );
        SMALL_PUSH_BACK( FLOAT    );
        LARGE_PUSH_BACK( STRLIT   );
        LARGE_PUSH_BACK( IDN      );

        size_t TokenArray::size() const
        {
            return off_len;
        }

#ifdef _DEBUG
        void TokenArray::print() const
        {
            for (int i = 0; i < size(); i++)
                std::cout << to_string((*this)[i]) << std::endl;
        }
#endif

        void lex_buf(char* buf, size_t bufsz, TokenArray& tarr)
        {
            uint32_t line_num = 1;
            uint32_t line_start = 0;
            bool use_indents = true;

            for (uint32_t i = 0; i < bufsz;)
            {
                switch (buf[i])
                {
                case ' ':
                    if (use_indents
                        && bufsz - i >= 4
                        && buf[i + 1] == ' '
                        && buf[i + 2] == ' '
                        && buf[i + 3] == ' '
                        )
                    {
                        tarr.push_back(TokenImp<TokenType::INDENT>(line_num, i - line_start));
                        i += 4;
                        continue;
                    }
                    break;
                case '\t':
                    if (use_indents)
                        tarr.push_back(TokenImp<TokenType::INDENT>(line_num, i - line_start));
                    break;
                case '\r':
                case '\v':
                case '\f':
                    break;
                case '\n':
                    use_indents = true;
                    line_start = i;
                    line_num++;
                    tarr.push_back(TokenImp<TokenType::ENDL>(line_num, i - line_start));
                    break;
                case '<':
                    use_indents = false;
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_LE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ANGLE_O>(line_num, i - line_start));
                    break;
                case '>':
                    use_indents = false;
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_GE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ANGLE_C>(line_num, i - line_start));
                    break;
                case '(':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::ROUND_O>(line_num, i - line_start));
                    break;
                case ')':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::ROUND_C>(line_num, i - line_start));
                    break;
                case '[':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::SQUARE_O>(line_num, i - line_start));
                    break;
                case ']':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::SQUARE_C>(line_num, i - line_start));
                    break;
                case '.':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::DOT>(line_num, i - line_start));
                    break;
                case ':':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::COLON>(line_num, i - line_start));
                    break;
                case ',':
                    use_indents = false;
                    tarr.push_back(TokenImp<TokenType::COMMA>(line_num, i - line_start));
                    break;
                case '+':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IADD>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ADD>(line_num, i - line_start));
                    break;
                case '*':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IMUL>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::STAR>(line_num, i - line_start));
                    break;
                case '/':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IDIV>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::DIV>(line_num, i - line_start));
                    break;
                case '!':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_NE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    throw SyntaxError(line_num, i - line_start, "Expected '=' after '!'");
                case '=':
                    use_indents = false;

                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_EQ>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ASSIGN>(line_num, i - line_start));
                    break;
                case '"':
                {
                    use_indents = false;

                    TokenImp<TokenType::STRLIT> tk(line_num, i - line_start);
                    int sidx = 0;
                    for (i += 1; i < bufsz && sidx < 256 && buf[i] != '"'; i++, sidx++)
                        tk.val[sidx] = buf[i];
                    if (i >= bufsz)
                        throw SyntaxError(tk.line_num, tk.col_num, "Missing closing '\"' for string literal");
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
                        if (bufsz - i >= 2 && buf[i + 1] == '=')
                        {
                            tarr.push_back(TokenImp<TokenType::ISUB>(line_num, col_num));
                            i += 2;
                            continue;
                        }
                        i += 1;
                        if (bufsz == i || (!is_numeric(buf[i]) && buf[i] != '.'))  // not a number
                        {
                            tarr.push_back(TokenImp<TokenType::SUB>(line_num, col_num));
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
                        throw SyntaxError(line_num, col_num, "Unexpected EOF while lexing integral type");
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
                                TokenImp<TokenType::INT> tk(line_num, col_num);
                                tk.val = ival;
                                tarr.push_back(tk);
                                continue;
                            }
                        }

                        // use ival as the >1 portion of the float, and find the <1 portion
                        float multiplier = 0.1;
                        double fval = ival;
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
                        TokenImp<TokenType::FLOAT> tk(line_num, col_num);
                        tk.val = fval;
                        tarr.push_back(tk);
                        continue;
                    }

                    // Only indentifiers are left
                    if (!is_idnstart(buf[i]))
                        throw SyntaxError(line_num, col_num, "Unexpected characted '{}'", buf[i]);

                    TokenImp<TokenType::IDN> tk(line_num, col_num);
                    int iidx = 0;
                    for (; i < bufsz && iidx < 64 && is_idnchar(buf[i]); i++, iidx++)
                        tk.val[iidx] = buf[i];
                    if (iidx == 64)
                        throw std::overflow_error("buffer overflow for identifier during lexing");
                    tk.val[iidx] = '\0';
                    tarr.push_back(tk);
                    continue;
                }
                }

                i++;
            }
        }

        void lex_file(FILE* pf, TokenArray& tarr)
        {
            // temp, bad implmentation
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
            lex_buf(pbuf, fsz, tarr);
            delete[] pbuf;
        }
    }
}
