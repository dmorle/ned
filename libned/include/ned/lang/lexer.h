#ifndef NED_LEXER_H
#define NED_LEXER_H

#include <vector>
#include <string>
#if __cplusplus >= 202002L
#include <format>
#else
namespace std {
    template<class... Args>
    string format(const string& fmt, const Args&...) { return fmt; }
}
#endif

#define FNAME_SIZE 256
#define IDN_SIZE   64
#define STR_SIZE   1024

namespace nn
{
    namespace lang
    {
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
            MOD,
            IADD,
            ISUB,
            IMUL,
            IDIV,
            IMOD,
            ASSIGN,
            CMP_EQ,
            CMP_NE,
            CMP_GT,
            CMP_LT,
            CMP_GE,
            CMP_LE,
            LIT_INT,
            LIT_FLOAT,
            LIT_STR,
            IDN,
            KW_STRUCT,
            KW_DEF,
            KW_INTR,
            KW_FN,
            KW_RETURN,
            KW_IMPORT,
            KW_WHILE,
            KW_FOR,
            KW_IN,
            KW_BREAK,
            KW_CONTINUE,
            KW_IF,
            KW_ELIF,
            KW_ELSE,
            KW_TYPE,
            KW_VAR,
            KW_FPTYPE,
            KW_BOOL,
            KW_INT,
            KW_FLOAT,
            KW_STR,
            KW_ARRAY,
            KW_TUPLE,
            KW_TENSOR,
            KW_TRUE,
            KW_FALSE,
            KW_RAISE,
            KW_EXPORT,
            KW_EXTERN,
            KW_F16,
            KW_F32,
            KW_F64,
            KW_PRINT,
            KW_INSTOF,
            KW_TYPEOF,
            KW_AND,
            KW_OR
        };

        template<TokenType T>
        class TokenImp;

        class Token
        {
        public:
            TokenType ty = TokenType::INVALID;
            char fname[FNAME_SIZE];
            uint32_t sz;
            uint32_t line_num;
            uint32_t col_num;

            Token(const char* fname, uint32_t sz, uint32_t line_num, uint32_t col_num) :
                sz(sz),
                line_num(line_num),
                col_num(col_num)
            {
                strncpy(this->fname, fname, 256);
            }

            // useful utilities

            bool is_whitespace() const { return ty == TokenType::INDENT || ty == TokenType::ENDL; }
            template<TokenType TY> TokenImp<TY>& get() {
                if (ty != TY) throw SyntaxError(this, "Expected {}, found {}", to_string(ty), to_string(TY));
                return *reinterpret_cast<TokenImp<TY>*>(this); }
            template<TokenType TY> const TokenImp<TY>& get() const {
                if (ty != TY) throw SyntaxError(this, "Expected {}, found {}", to_string(ty), to_string(TY));
                return *reinterpret_cast<const TokenImp<TY>*>(this); }
        };

        constexpr std::string to_string(const TokenType ty);
        std::string to_string(const Token* ptk);

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
            SyntaxError(const Token* ptoken, const std::string& fmt, const Args&... args)
            {
                this->line_num = ptoken->line_num;
                this->col_num = ptoken->col_num;
                this->errmsg = std::format(fmt, args...);
            }

            template<typename... Args>
            SyntaxError(const Token& token, const std::string& fmt, const Args&... args)
            {
                this->line_num = token.line_num;
                this->col_num = token.col_num;
                this->errmsg = std::format(fmt, args...);
            }

            virtual char const* what() const
            {
                return errmsg.c_str();
            }
        };

        template<TokenType T>
        class TokenImp :
            public Token
        {
        public:
            TokenImp(const char* fname, uint32_t line_num, uint32_t col_num) :
                Token(fname, sizeof(TokenImp<T>), line_num, col_num)
            {
                ty = T;
            }
        };

        template<>
        class TokenImp<TokenType::LIT_INT> :
            public Token
        {
        public:
            int64_t val;

            TokenImp(const char* fname, uint32_t line_num, uint32_t col_num) :
                Token(fname, sizeof(TokenImp<TokenType::LIT_INT>), line_num, col_num)
            {
                ty = TokenType::LIT_INT;
                val = 0;
            }
        };

        template<>
        class TokenImp<TokenType::LIT_FLOAT> :
            public Token
        {
        public:
            double val;

            TokenImp(const char* fname, uint32_t line_num, uint32_t col_num) :
                Token(fname, sizeof(TokenImp<TokenType::LIT_FLOAT>), line_num, col_num)
            {
                ty = TokenType::LIT_FLOAT;
                val = 1.0;
            }
        };

        template<>
        class TokenImp<TokenType::LIT_STR> :
            public Token
        {
        public:
            char val[STR_SIZE];

            TokenImp(const char* fname, uint32_t line_num, uint32_t col_num) :
                Token(fname, sizeof(TokenImp<TokenType::LIT_STR>), line_num, col_num)
            {
                ty = TokenType::LIT_STR;
                val[0] = '\0';
            }
        };

        template<>
        class TokenImp<TokenType::IDN> :
            public Token
        {
        public:
            char val[IDN_SIZE];

            TokenImp(const char* fname, uint32_t line_num, uint32_t col_num) :
                Token(fname, sizeof(TokenImp<TokenType::IDN>), line_num, col_num)
            {
                ty = TokenType::IDN;
                val[0] = '\0';
            }
        };

        class TokenArrayIterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = Token;
            using pointer           = value_type*;
            using reference         = value_type&;

            TokenArrayIterator(pointer ptr) : ptr(ptr) {}

            reference operator*() const { return *ptr; }
            pointer operator->() { return ptr; }

            TokenArrayIterator& operator++() { ptr = (pointer)(((uint8_t*)ptr) + ptr->sz); return *this; }
            TokenArrayIterator operator++(int) { TokenArrayIterator tmp = *this; ++(*this); return tmp; }

            friend bool operator== (const TokenArrayIterator& a, const TokenArrayIterator& b) { return a.ptr == b.ptr; };
            friend bool operator!= (const TokenArrayIterator& a, const TokenArrayIterator& b) { return a.ptr != b.ptr; };

        private:
            pointer ptr;
        };

        class TokenArrayConstIterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = Token;
            using pointer = const value_type*;
            using reference = const value_type&;

            TokenArrayConstIterator(const pointer ptr) : ptr(ptr) {}

            reference operator*() const { return *ptr; }
            pointer operator->() { return ptr; }

            TokenArrayConstIterator& operator++() { ptr = (pointer)(((uint8_t*)ptr) + ptr->sz); return *this; }
            TokenArrayConstIterator operator++(int) { TokenArrayConstIterator tmp = *this; ++(*this); return tmp; }

            friend bool operator== (const TokenArrayConstIterator& a, const TokenArrayConstIterator& b) { return a.ptr == b.ptr; };
            friend bool operator!= (const TokenArrayConstIterator& a, const TokenArrayConstIterator& b) { return a.ptr != b.ptr; };

        private:
            pointer ptr;
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
            
            TokenArrayIterator begin() { return TokenArrayIterator((Token*)pbuf); }
            TokenArrayIterator end() { uint8_t* last = pbuf + offsets[off_len - 1];
                return TokenArrayIterator(reinterpret_cast<Token*>(last + reinterpret_cast<Token*>(last)->sz)); }
            TokenArrayConstIterator begin() const { return TokenArrayConstIterator((Token*)pbuf); }
            TokenArrayConstIterator end() const { const uint8_t* last = pbuf + offsets[off_len - 1];
                return TokenArrayConstIterator(reinterpret_cast<const Token*>(last + reinterpret_cast<const Token*>(last)->sz)); }
            const Token* operator[](size_t idx) const noexcept;
            size_t size() const;

            template<TokenType T>
            void push_back(const TokenImp<T>& tk)
            {
                constexpr size_t nsz = sizeof(TokenImp<T>);
                if (is_slice)
                    throw std::runtime_error("TokenArray slices are immutable");
                void* tmp;
                while (rawlen + nsz >= mem_sz)
                {
                    mem_sz += mem_sz;
                    tmp = std::realloc(pbuf, mem_sz);
                    if (!tmp)
                        throw std::bad_alloc();
                    pbuf = (uint8_t*)tmp;
                }
                off_len++;
                if (off_len == off_cap)
                {
                    off_cap += off_cap;
                    tmp = std::realloc(offsets, sizeof(size_t) * off_cap);
                    if (!tmp)
                        throw std::bad_alloc();
                    offsets = (size_t*)tmp;
                }
                offsets[off_len - 1] = rawlen;
                std::memcpy(pbuf + rawlen, &tk, nsz);
                rawlen += nsz;
            }

#ifdef _DEBUG
            void print() const;
#endif

            // returns -1 if token is not present in range
            template<class SearchCriteria>
            int search(SearchCriteria&& sc, int start = 0) const
            {
                // A bit better cache utilization than rfind
                int idx = start;
                const Token* pstart = reinterpret_cast<const Token*>(this->pbuf + this->offsets[start]);
                while (idx < size())
                {
                    int ret = sc.accept(pstart, idx);
                    if (ret >= 0)
                        return ret;
                    idx++;
                    pstart = reinterpret_cast<const Token*>((const uint8_t*)pstart + pstart->sz);
                }

                return -1;
            }

            // returns -1 if token is not present in range
            // less efficient than a forward search
            // start argument < 0 offsets from the end of the token array
            template<class SearchCriteria>
            int rsearch(SearchCriteria&& sc, int start = -1, int end = 0) const
            {
                for (int i = start % this->off_len; i >= end; i--)
                {
                    int ret = sc.accept(reinterpret_cast<const Token*>(this->pbuf + this->offsets[i]), i);
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

        void lex_buf(const char* fname, char* buf, size_t bufsz, TokenArray& tarr);
        void lex_file(const char* fname, FILE* pf, TokenArray& tarr);

        class BracketCounter
        {
        protected:
            size_t rbrac = 0;  // Round  brackets ()
            size_t sbrac = 0;  // Square brackets []
            size_t abrac = 0;  // Angled brackets <>

            void count_token(const Token* ptk);  // Returns
            bool in_bracket() const;
        };

        class IsSameCriteria :
            public BracketCounter
        {
            TokenType ty;
        public:
            IsSameCriteria(TokenType ty);
            int accept(const Token* ptk, int idx);
        };

        class CargEndCriteria :
            public BracketCounter
        {
        public:
            int accept(const Token* ptk, int idx);
        };

        class VargEndCriteria :
            public BracketCounter
        {
        public:
            int accept(const Token* ptk, int idx);
        };

        /*
        * Finds the position of the line end after a colon which indicates the start of a block
        * 
        * Ex.
        *   def add<>():
        *       return
        * Given the TokenArray as follows:
        *   def
        *   add
        *   <
        *   >
        *   (
        *   )
        *   :
        *   endl  <- The criteria will return the position of this token
        *   tab
        *   return
        *   endl
        */
        class BlockStartCriteria :
            public BracketCounter
        {
        public:
            int accept(const Token* ptk, int idx);
        };

        /*
        * Finds the position of the line end after the last non-whitespace token in the line
        * 
        * Ex1.
        *   x = x
        * Given the TokenArray as follows:
        *   x
        *   =
        *   x
        *   endl  <- The criteria will return the position of this token
        * 
        * Ex2.
        *       if (
        *           true):
        *           print "true"
        *       return
        * Given the TokenArray as follows:
        *   tab
        *   if
        *   (
        *   endl
        *   tab
        *   tab
        *   true
        *   )
        *   :
        *   endl
        *   tab
        *   tab
        *   print
        *   "true"
        *   endl        <- The criteria will return the position of this token
        *   tab
        *   return
        *   endl
        */
        class LineEndCriteria :
            public BracketCounter
        {
            int target_ilv;
            int current_ilv = 0;
            int last_endl = -1;

        public:
            LineEndCriteria(int indent_level);

            int accept(const Token* ptk, int idx);
        };
    }
}

#endif
