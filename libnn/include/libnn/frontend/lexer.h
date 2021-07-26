#ifndef NN_LEXER_H
#define NN_LEXER_H

#include <vector>
#include <string>
#include <format>

namespace nn
{
    namespace frontend
    {
        class SyntaxError :
            public std::exception
        {
        public:
            std::string errmsg;
            size_t line_num;
            size_t col_num;

            SyntaxError(size_t line_num, size_t col_num, const std::string& fmt)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = fmt;
            }

            template<typename... Args>
            SyntaxError(size_t line_num, size_t col_num, const std::string& fmt, const Args&... args)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            SyntaxError(Token* ptoken, const std::string& fmt, const Args&... args)
            {
                this->line_num = ptoken->line_num;
                this->col_num = ptoken->col_num;
                this->errmsg = std::format(fmt, args);
            }

            template<typename... Args>
            SyntaxError(Token& token, const std::string& fmt, const Args&... args)
            {
                this->line_num = token.line_num;
                this->col_num = token.col_num;
                this->errmsg = std::format(fmt, args);
            }

            virtual char const* what() const
            {
                return errmsg.c_str();
            }
        };

        enum class TokenType
        {
            INVALID,
            INDENT,
            ENDL,
            ANGLE_O,
            ANGLE_C,
            ROUND_O,
            ROUND_C,
            SQUARE_O,
            SQUARE_C,
            DOT,
            COLON,
            COMMA,
            ADD,
            SUB,
            STAR,
            DIV,
            IADD,
            ISUB,
            IMUL,
            IDIV,
            ASSIGN,
            CMP_EQ,
            CMP_NE,
            CMP_GE,
            CMP_LE,
            INT,
            FLOAT,
            STRLIT,
            IDN
        };

        enum class Keyword
        {
            DEF,
            IF,
            ELIF,
            ELSE,
            FOR,
            IN,
            WHILE,
            VAR,
            IMPORT,
            RETURN
        };

        class Token
        {
        public:
            TokenType ty = TokenType::INVALID;
            uint32_t sz;
            uint32_t line_num;
            uint32_t col_num;

            Token(uint32_t sz, uint32_t line_num, uint32_t col_num) :
                sz(sz),
                line_num(line_num),
                col_num(col_num)
            {}
        };

        template<TokenType T>
        class TokenImp :
            public Token
        {
        public:
            TokenImp(uint32_t line_num, uint32_t col_num) :
                Token(sizeof(TokenImp<T>), line_num, col_num)
            {
                ty = T;
            }
        };

        template<>
        class TokenImp<TokenType::INT> :
            public Token
        {
        public:
            int64_t val;

            TokenImp(uint32_t line_num, uint32_t col_num) :
                Token(sizeof(TokenImp<TokenType::INT>), line_num, col_num)
            {
                ty = TokenType::INT;
                val = 0;
            }
        };

        template<>
        class TokenImp<TokenType::FLOAT> :
            public Token
        {
        public:
            double val;

            TokenImp(uint32_t line_num, uint32_t col_num) :
                Token(sizeof(TokenImp<TokenType::FLOAT>), line_num, col_num)
            {
                ty = TokenType::FLOAT;
                val = 1.0;
            }
        };

        template<>
        class TokenImp<TokenType::STRLIT> :
            public Token
        {
        public:
            char val[256];

            TokenImp(uint32_t line_num, uint32_t col_num) :
                Token(sizeof(TokenImp<TokenType::STRLIT>), line_num, col_num)
            {
                ty = TokenType::STRLIT;
                val[0] = '\0';
            }
        };

        template<>
        class TokenImp<TokenType::IDN> :
            public Token
        {
        public:
            char val[64];

            TokenImp(uint32_t line_num, uint32_t col_num) :
                Token(sizeof(TokenImp<TokenType::IDN>), line_num, col_num)
            {
                ty = TokenType::IDN;
                val[0] = '\0';
            }
        };

        class TokenArray
        {
        public:
            TokenArray(size_t mem_sz=512, size_t off_cap=16);
            TokenArray(const TokenArray& base, int start = 0);  // slice constructor
            TokenArray(const TokenArray& base, int start, int end);  // slice constructor
            ~TokenArray();

            TokenArray(TokenArray&&) noexcept;
            TokenArray(const TokenArray&) = delete;
            TokenArray& operator=(TokenArray&&) noexcept;
            TokenArray& operator=(const TokenArray&) = delete;
            
            const Token* operator[](size_t idx) const noexcept;
            template<TokenType T> void push_back(const TokenImp<T>& tk);
            size_t size() const;

            template<TokenType Ty>
            static bool is_same(const TokenType ty)
            {
                return ty == Ty;
            }

            template<TokenType Ty>
            static bool is_same_brac(const TokenType ty)
            {
                static int rbrac = 0;
                static int sbrac = 0;
                static int abrac = 0;

                switch (ty)
                {
                case TokenType::ROUND_O:
                    rbrac++;
                    return false;
                case TokenType::ROUND_C:
                    rbrac--;
                    return false;
                case TokenType::SQUARE_O:
                    sbrac++;
                    return false;
                case TokenType::SQUARE_C:
                    sbrac--;
                    return false;
                case TokenType::ANGLE_O:
                    abrac++;
                    return false;
                case TokenType::ANGLE_C:
                    abrac--;
                    return false;
                case Ty:
                    return !(rbrac || sbrac || abrac);
                default:
                    return false;
                }
            }

            template<Keyword kw>
            static bool is_keyword(const Token* tk);
            template<> static bool is_keyword<Keyword::DEF>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "def"; }
            template<> static bool is_keyword<Keyword::IF>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "if"; }
            template<> static bool is_keyword<Keyword::ELIF>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "elif"; }
            template<> static bool is_keyword<Keyword::ELSE>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "else"; }
            template<> static bool is_keyword<Keyword::FOR>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "for"; }
            template<> static bool is_keyword<Keyword::IN>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "in"; }
            template<> static bool is_keyword<Keyword::WHILE>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "while"; }
            template<> static bool is_keyword<Keyword::VAR>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "var"; }
            template<> static bool is_keyword<Keyword::IMPORT>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "import"; }
            template<> static bool is_keyword<Keyword::RETURN>(const Token* tk)
            { return tk->ty == TokenType::IDN && static_cast<const TokenImp<TokenType::IDN>*>(tk)->val == "return"; }

            template<int ILV>
            static int block_end(const TokenType ty, int position)
            {
                static int ilv = 0;
                static int last_endl = 0;
                static int in_line = false;

                if (ty == TokenType::ENDL)
                {
                    last_endl = position;
                    ilv = 0;
                    in_line = false;
                    return -1;
                }
                if (in_line)
                    return -1;
                if (ty == TokenType::INDENT)
                {
                    ilv++;
                    return -1;
                }
                if (ilv >= ILV)
                    return -1;

                int ret = last_endl + 1;
                ilv = 0;
                last_endl = 0;
                return ret;
            }

            template<TokenType OPEN, TokenType CLOSE>
            static bool brac_end(const TokenType ty)
            {
                static int lvl = 0;

                if (ty == TokenType::CLOSE)
                {
                    if (lvl == 0)
                        return true;
                    lvl--;
                    return false;
                }
                if (ty == TokenType::OPEN)
                    lvl++;
                return false;
            }

            template<TokenType OPEN, TokenType CLOSE>
            static bool args_elem(const TokenType ty)
            {
                static int lvl = 0;

                if (ty == CLOSE)
                {
                    if (lvl == 0)
                        return true;
                    lvl--;
                    return false;
                }
                if (ty == OPEN)
                {
                    lvl++;
                    return false;
                }

                return ty == TokenType::COMMA && lvl == 0;
            }

            // returns -1 if token is not present in range
            // start/end arguments <0 offsets from the end of the token array
            template<bool(*F)(TokenType)>
            int search(int start = 0) const
            {
                // A bit better cache utilization than rfind
                int result = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (result < size())
                {
                    if (F(pstart->ty))
                        return result;
                    result++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<bool(*F)(TokenType)>
            int search(int start, int end) const
            {
                end %= this->off_len;

                // A bit better cache utilization than rfind
                int result = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (result < end)
                {
                    if (F(pstart->ty))
                        return result;
                    result++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<bool(*F)(const Token*)>
            int search(int start = 0) const
            {
                // A bit better cache utilization than rfind
                int result = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (result < size())
                {
                    if (F(pstart))
                        return result;
                    result++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<bool(*F)(const Token*)>
            int search(int start, int end) const
            {
                end %= this->off_len;

                // A bit better cache utilization than rfind
                int result = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (result < end)
                {
                    if (F(pstart))
                        return result;
                    result++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<int(*F)(TokenType, int)>
            int search(int start = 0) const
            {
                // A bit better cache utilization than rfind
                int ret;
                int idx = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (idx < end)
                {
                    ret = F(pstart->ty, idx);
                    if (ret >= 0)
                        return ret;
                    idx++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<int(*F)(TokenType, int)>
            int search(int start, int end) const
            {
                end %= this->off_len;

                // A bit better cache utilization than rfind
                int ret;
                int idx = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (idx < end)
                {
                    ret = F(pstart->ty, idx);
                    if (ret >= 0)
                        return ret;
                    idx++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<int(*F)(const Token*, int)>
            int search(int start = 0) const
            {
                // A bit better cache utilization than rfind
                int ret;
                int idx = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (idx < end)
                {
                    ret = F(pstart, idx);
                    if (ret >= 0)
                        return ret;
                    idx++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            template<int(*F)(const Token*, int)>
            int search(int start, int end) const
            {
                end %= this->off_len;

                // A bit better cache utilization than rfind
                int ret;
                int idx = start;
                const Token* pstart = static_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (idx < end)
                {
                    ret = F(pstart, idx);
                    if (ret >= 0)
                        return ret;
                    idx++;
                    pstart = static_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            // less efficient than a forward search
            // returns -1 if token is not present in range
            // start/end arguments <0 offsets from the end of the token array
            template<bool(*F)(TokenType)>
            int rsearch(int start = -1, int end = 0) const
            {
                end %= this->off_len;

                for (int i = start % this->off_len; i >= end; i--)
                    if (F(static_cast<const Token*>(this->pbuf + this->offsets[i])->ty))
                        return i;
                return -1;
            }

            template<bool(*F)(const Token*)>
            int rsearch(int start = -1, int end = 0) const
            {
                end %= this->off_len;

                for (int i = start % this->off_len; i >= end; i--)
                    if (F(static_cast<const Token*>(this->pbuf + this->offsets[i])))
                        return i;
                return -1;
            }

            template<int(*F)(TokenType, int)>
            int rsearch(int start = -1, int end = 0) const
            {
                start %= this->off_len;

                int ret;
                for (int i = start % this->off_len; i >= end; i--)
                {
                    ret = F(static_cast<const Token*>(this->pbuf + this->offsets[i])->ty, i);
                    if (ret >= 0)
                        return ret;
                }
                return -1;
            }

            template<int(*F)(const Token*, int)>
            int rsearch(int start = -1, int end = 0) const
            {
                start %= this->off_len;

                int ret;
                for (int i = start % this->off_len; i >= end; i--)
                {
                    ret = F(static_cast<const Token*>(this->pbuf + this->offsets[i]), i);
                    if (ret >= 0)
                        return ret;
                }
                return -1;
            }

        private:
            bool is_slice = false;

            size_t mem_sz = 0;
            size_t rawlen = 0;
            uint8_t* pbuf = nullptr;
            
            size_t off_cap = 0;
            size_t off_len = 0;
            size_t* offsets = nullptr;
        };

        TokenArray lex_buf(char* buf, size_t bufsz);
        TokenArray lex_file(FILE* pf);
    }
}

#endif
