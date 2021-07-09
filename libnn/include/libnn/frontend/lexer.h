#ifndef NN_LEXER_H
#define NN_LEXER_H

#include <vector>
#include <string>
#include <format>

#include <libnn/frontend/parser.h>

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

            SyntaxError(size_t line_num, size_t col_num, std::string fmt)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = fmt;
            }

            template<typename... Args>
            SyntaxError(size_t line_num, size_t col_num, std::string fmt, const Args&... args)
            {
                this->line_num = line_num;
                this->col_num = col_num;
                this->errmsg = std::format(fmt, args...);
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
            ANGLE_O,
            ANGLE_C,
            ROUND_O,
            ROUND_C,
            SQUARE_O,
            SQUARE_C,
            DOT,
            COMMA,
            ADD,
            SUB,
            MUL,
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

        class Token
        {
        public:
            TokenType ty = TokenType::INVALID;
            uint32_t line_num;
            uint32_t col_num;

            Token(uint32_t line_num, uint32_t col_num) :
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
                Token(line_num, col_num)
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
                Token(line_num, col_num)
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
                Token(line_num, col_num)
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
                Token(line_num, col_num)
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
                Token(line_num, col_num)
            {
                ty = TokenType::IDN;
                val[0] = '\0';
            }
        };

        class TokenArray
        {
        public:
            TokenArray(size_t mem_sz=512, size_t off_cap=16);
            TokenArray(const TokenArray& base, size_t start, size_t end);  // slice constructor
            ~TokenArray();

            TokenArray(TokenArray&&) noexcept;
            TokenArray(const TokenArray&) = delete;
            TokenArray& operator=(TokenArray&&) noexcept;
            TokenArray& operator=(const TokenArray&) = delete;
            
            const Token* operator[](size_t idx) const noexcept;
            template<TokenType T> void push_back(const TokenImp<T>& tk);

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
