#ifndef NED_BYTECODE_H
#define NED_BYTECODE_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <type_traits>

#include <ned/lang/lexer.h>
#include <ned/lang/obj.h>
#include <ned/lang/interp.h>

namespace nn
{
    namespace lang
    {
        class Instruction
        {
        protected:
            std::string fname;
            size_t line_num, col_num;

            Instruction(const Token* ptk) : fname(ptk->fname), line_num(ptk->line_num), col_num(ptk->col_num) {}
        public:
            virtual CodeSegPtr to_bytes(CodeSegPtr buf) const = 0;
            virtual bool set_labels(Errors& errs, const std::map<std::string, size_t>& label_map) = 0;
        };
        
        class ByteCodeBody
        {
            std::vector<std::unique_ptr<Instruction>> instructions;
            std::map<std::string, size_t> label_map;
            size_t body_sz;

        public:
            size_t size() const;
            CodeSegPtr to_bytes(Errors& errs, size_t offset, CodeSegPtr buf) const;
            bool add_label(Errors& errs, const TokenImp<TokenType::IDN>* lbl);

            template<typename T>
            bool add_instruction(const T& inst)
            {
                static_assert(std::is_same<decltype(T::value), size_t>::value);
                body_sz += T::size;
                instructions.push_back(std::make_unique(inst));
                return false;
            }
        };

        class ByteCodeModule
        {
            std::map<std::string, ByteCodeBody> blocks;
            std::vector<Obj> statics;
            ProgramHeap& heap;

            friend bool parsebc_static(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, Obj& obj);
        public:
            ByteCodeModule(ProgramHeap& heap) : heap(heap) {}
            bool add_block(Errors& errs, const std::string& name, const ByteCodeBody& body);
            bool add_static(Obj obj, size_t& addr);
            bool export_module(CodeSegPtr& code_segment, DataSegPtr& data_segment);
        };

        enum class InstructionType : uint8_t
        {
            JMP,
            BRT,
            BRF,
            NEW,
            AGG,
            ARR,
            ATY,
            POP,
            DUP,
            CPY,
            INST,
            CALL,
            RET,
            SET,
            IADD,
            ISUB,
            IMUL,
            IDIV,
            IMOD,
            ADD,
            SUB,
            MUL,
            DIV,
            MOD,
            EQ,
            NE,
            GT,
            LT,
            GE,
            LE,
            IDX,
            XSTR,
            XFLT,
            XINT,

            TEN,
            EBLK,
            DBLK,
            CBLK,
            IBLK,
            BINP,
            BOUT,
            EXT,
            EXP
        };

        namespace instruction
        {
            template<InstructionType OPCODE>
            class Labeled :  // instructions of the form <opcode> <label>
                public Instruction
            {
                std::string label;
                size_t label_ptr = 0;
            public:
                Labeled(const Token* ptk, std::string label) : Instruction(ptk), label(label) {}
                static constexpr size_t size = sizeof(InstructionType) + sizeof(size_t);

                virtual bool set_labels(Errors& errs, const std::map<std::string, size_t>& label_map) override
                {
                    if (!label_map.contains(label))
                        return errs.add(line_num, col_num, "Unresolved reference to label '{}'", label);
                    label_ptr = label_map.at(label);
                    return false;
                }

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *static_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    *static_cast<size_t*>(buf) = label_ptr;
                    buf += sizeof(size_t);
                    return buf;
                }
            };

            template<InstructionType OPCODE>
            class Valued :  // instructions of the form <opcode> <uint>
                public Instruction
            {
                size_t val;
            public:
                Valued(const Token* ptk, size_t val) : Instruction(ptk), val(val) {}
                static constexpr size_t size = sizeof(InstructionType) + sizeof(size_t);

                virtual bool set_labels(Errors&, const std::map<std::string, size_t>&) override {}

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *static_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    *static_cast<size_t*>(buf) = val;
                    buf += sizeof(size_t);
                    return buf;
                }
            };

            template<InstructionType OPCODE>
            class Implicit :  // instructions of the form <opcode>
                public Instruction
            {
            public:
                Implicit(const Token* ptk) : Instruction(ptk) {}
                static constexpr size_t size = sizeof(InstructionType);

                virtual bool set_labels(Errors&, const std::map<std::string, size_t>&) override {}

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *static_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    return buf;
                }
            };

            using enum ::nn::lang::InstructionType;
            using Jmp  = Labeled  < JMP  >;
            using Brt  = Labeled  < BRT  >;
            using Brf  = Labeled  < BRF  >;
            using New  = Valued   < NEW  >;
            using Agg  = Valued   < AGG  >;
            using Arr  = Implicit < ARR  >;
            using Aty  = Valued   < ATY  >;
            using Pop  = Valued   < POP  >;
            using Dup  = Valued   < DUP  >;
            using Cpy  = Implicit < CPY  >;
            using Inst = Implicit < INST >;
            using Call = Implicit < CALL >;
            using Ret  = Implicit < RET  >;
            using Set  = Implicit < SET  >;
            using IAdd = Implicit < IADD >;
            using ISub = Implicit < ISUB >;
            using IMul = Implicit < IMUL >;
            using IDiv = Implicit < IDIV >;
            using IMod = Implicit < IMOD >;
            using Add  = Implicit < ADD  >;
            using Sub  = Implicit < SUB  >;
            using Mul  = Implicit < MUL  >;
            using Div  = Implicit < DIV  >;
            using Mod  = Implicit < MOD  >;
            using Eq   = Implicit < EQ   >;
            using Ne   = Implicit < NE   >;
            using Ge   = Implicit < GE   >;
            using Le   = Implicit < LE   >;
            using Gt   = Implicit < GT   >;
            using Lt   = Implicit < LT   >;
            using Idx  = Implicit < IDX  >;
            using XStr = Implicit < XSTR >;
            using XFlt = Implicit < XFLT >;
            using XInt = Implicit < XINT >;
        }
        
        bool parsebc_static(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, size_t& addr);
        bool parsebc_instruction(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body);
        bool parsebc_body(Errors& errs, const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body);
        bool parsebc_module(Errors& errs, const char* fname, char* buf, size_t bufsz, ByteCodeModule& mod);
        bool parsebc_module(Errors& errs, const char* fname, FILE* pf, ByteCodeModule& mod);
    }
}

/*
* Language Instructions
* 
* jmp <label>  Unconditional Jump
* brt <label>  Branch if true with pop
* brf <label>  Branch if false with pop
* 
* new <addr>   Adds an object from static memory onto the stack
* agg <uint>   Creates a new aggregate object from the top <uint> elements on the stack
* arr          Creates a new array type with element type specified by the tos (used for generics)
* aty <uint>   Creates a new struct type from the top <uint> elements on the stack (used for generics)
* 
* pop <uint>   Pops element <uint> off the stack
* dup <uint>   Duplicates an object on the stack
* cpy          Copies the object on the tos
* inst         Creates an instance from the element on the top of the stack, and pops the type
* 
* Considers the top element on the stack to be a pointer into the code segment
* Pops the tos, pushes the current pc onto the stack and sets a new pc
* call
* 
* Returns <uint> elements from a code block
* Retrieves the program counter and a reference to the caller from the call stack
* Relies on the caller to cleanup the stack from the call, and updates the program counter
* ret
* 
* Binary Operations; Pops the top two elements off the stack, does the op, pushes the result
* 
* set          Assigns the obj at tos-1 to tos
* iadd         Add and assigns
* isub         Subtract and assign
* imul         Multiply and assign
* idiv         Divide and assign
* imod         Modulus and assign
* add          Adds two objects
* sub          Subtracts two objects
* mul          Multiplies two objects
* div          Divides two objects
* mod          Calculates the modulus between two objects
* 
* eq           Checks if two objects are equal
* ne           Checks if two objects are not equal
* ge           Checks if the left is greater than or equal to the right
* le           Checks if the left if less than or equal to the right
* gt           Checks if the left is greater than the right
* lt           Checks if the left is less than the right
* 
* idx          Retrieves the left at the index of the right (does not need type info)
* 
* xstr         Converts any object to a string object
* xflt         Converts any object to a float object
* xint         Converts any object to an int object
* 
* 
* Deep learning instructions
* 
* ten <uint>   Creates a new tensor with rank <uint>
* 
* eblk         Enable deep learning instructions
* dblk         Disable deep learning instructions
* 
* cblk         Creates a new counpound block context with a name
* iblk         Creates a new intrinsic block context with a name
* binp         Marks a tensor as a block input with a name
* bout         Marks a tensor as a block output with a name
* 
* ext          Marks a tensor as a model weight
* exp          Exports a tensor with a name
* 
*/

#endif
