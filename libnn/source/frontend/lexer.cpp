#include <libnn/frontend/lexer.h>

#include <stdexcept>


namespace nn
{
    namespace frontend
    {
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

        TokenArray::TokenArray(const TokenArray& base, size_t start, size_t end)
        {
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
    offsets[off_len] = rawlen;                                                 \
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
    offsets[off_len] = rawlen;                                                 \
    std::memcpy(pbuf + rawlen, &tk, nsz);                                      \
    rawlen += nsz;                                                             \
}

        SMALL_PUSH_BACK( ANGLE_O  )
        SMALL_PUSH_BACK( ANGLE_C  )
        SMALL_PUSH_BACK( ROUND_O  )
        SMALL_PUSH_BACK( ROUND_C  )
        SMALL_PUSH_BACK( SQUARE_O )
        SMALL_PUSH_BACK( SQUARE_C )
        SMALL_PUSH_BACK( DOT      )
        SMALL_PUSH_BACK( COMMA    )
        SMALL_PUSH_BACK( ADD      )
        SMALL_PUSH_BACK( SUB      )
        SMALL_PUSH_BACK( MUL      )
        SMALL_PUSH_BACK( DIV      )
        SMALL_PUSH_BACK( IADD     )
        SMALL_PUSH_BACK( ISUB     )
        SMALL_PUSH_BACK( IMUL     )
        SMALL_PUSH_BACK( IMUL     )
        SMALL_PUSH_BACK( IMUL     )
        SMALL_PUSH_BACK( IDIV     )
        SMALL_PUSH_BACK( ASSIGN   )
        SMALL_PUSH_BACK( CMP_EQ   )
        SMALL_PUSH_BACK( CMP_NE   )
        SMALL_PUSH_BACK( CMP_GE   )
        SMALL_PUSH_BACK( CMP_LE   )
        SMALL_PUSH_BACK( BOOL     )
        SMALL_PUSH_BACK( INT      )
        SMALL_PUSH_BACK( FLOAT    )
        LARGE_PUSH_BACK( STRLIT   )
        LARGE_PUSH_BACK( IDN      )

        int lex_buf(char* buf, size_t bufsz, TokenArray& tarr)
        {
            size_t line_num = 1;
            size_t line_start = 0;

            for (size_t i = 0; i < bufsz;)
            {
                switch (buf[i])
                {
                case ' ':
                case '\t':
                case '\r':
                case '\v':
                case '\f':
                    break;
                case '\n':
                    line_start = i;
                    line_num++;
                    break;
                case '<':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_LE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ANGLE_O>(line_num, i - line_start));
                    break;
                case '>':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_GE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ANGLE_C>(line_num, i - line_start));
                    break;
                case '(':
                    tarr.push_back(TokenImp<TokenType::ROUND_O>(line_num, i - line_start));
                    break;
                case ')':
                    tarr.push_back(TokenImp<TokenType::ROUND_C>(line_num, i - line_start));
                    break;
                case '[':
                    tarr.push_back(TokenImp<TokenType::SQUARE_O>(line_num, i - line_start));
                    break;
                case ']':
                    tarr.push_back(TokenImp<TokenType::SQUARE_C>(line_num, i - line_start));
                    break;
                case '.':
                    tarr.push_back(TokenImp<TokenType::DOT>(line_num, i - line_start));
                    break;
                case ',':
                    tarr.push_back(TokenImp<TokenType::COMMA>(line_num, i - line_start));
                    break;
                case '+':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IADD>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ADD>(line_num, i - line_start));
                    break;
                case '-':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::ISUB>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::SUB>(line_num, i - line_start));
                    break;
                case '*':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IMUL>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::MUL>(line_num, i - line_start));
                    break;
                case '/':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::IDIV>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::DIV>(line_num, i - line_start));
                    break;
                case '!':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_NE>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    throw SyntaxError(line_num, i - line_start, "Expected '=' after '!'");
                case '=':
                    if (bufsz - i >= 2 && buf[i + 1] == '=')
                    {
                        tarr.push_back(TokenImp<TokenType::CMP_EQ>(line_num, i - line_start));
                        i += 2;
                        continue;
                    }
                    tarr.push_back(TokenImp<TokenType::ASSIGN>(line_num, i - line_start));
                    break;
                default:
                    break;
                }

                i++;
            }

            return 0;
        }

        int lex_file(FILE* pf, TokenArray& tarr)
        {
            // temp, bad implmentation
            fseek(pf, 0, SEEK_END);
            size_t fsz = ftell(pf);
            rewind(pf);
            char* pbuf = new char[fsz];
            size_t result = fread(pbuf, 1, fsz, pf);
            if (result != fsz)
            {
                delete[] pbuf;
                throw std::runtime_error("fread failed");
            }
            int ret = lex_buf(pbuf, fsz, tarr);
            delete[] pbuf;
            return ret;
        }
    }
}
