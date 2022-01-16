#ifndef NED_BYTECODE_H
#define NED_BYTECODE_H

#include <ned/errors.h>
#include <ned/lang/lexer.h>
#include <ned/lang/obj.h>
#include <ned/lang/interp.h>

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <type_traits>

namespace nn
{
    namespace lang
    {
        class ByteCodeDebugInfo
        {
        public:
            struct Record
            {
                size_t addr;
                std::string fname;
                size_t line_num, col_num;
            };

        private:
            std::vector<Record> instruction_records;
            friend class ByteCodeBody;

        public:
            const Record& at(size_t pc) const;
        };

        class Instruction
        {
        protected:
            std::string fname;
            size_t line_num, col_num;

            Instruction(const Token* ptk) : fname(ptk->fname), line_num(ptk->line_num), col_num(ptk->col_num) {}
        public:
            ByteCodeDebugInfo::Record create_debug_record(size_t addr) const;
            virtual CodeSegPtr to_bytes(CodeSegPtr buf) const = 0;
            virtual bool set_labels(const std::map<std::string, size_t>& label_map) = 0;
        };
        
        class ByteCodeBody
        {
            std::string fname;
            size_t line_num, col_num;
            friend class ByteCodeModule;

            std::vector<Instruction*> instructions;
            std::map<std::string, size_t> label_map;
            size_t body_sz;

        public:
            ByteCodeBody(const Token* ptk);
            size_t size() const;
            CodeSegPtr to_bytes(size_t offset, CodeSegPtr buf, ByteCodeDebugInfo& debug_info) const;
            bool add_label(const TokenImp<TokenType::IDN>* lbl);

            template<typename T>
            bool add_instruction(const T& inst)
            {
                body_sz += T::size;
                instructions.push_back(new T(inst));
                return false;
            }
        };

        struct ByteCode
        {
            CodeSegPtr code_segment;
            DataSegPtr data_segment;
            ProcOffsets proc_offsets;
            ByteCodeDebugInfo debug_info;
        };

        class ByteCodeModule
        {
            struct ProcRef
            {
                size_t addr;
                std::string proc_name;
                std::string fname;
                size_t line_num, col_num;
            };

            std::map<std::string, ByteCodeBody> procs;
            std::vector<Obj> statics;
            std::vector<ProcRef> static_proc_refs;
            ProgramHeap& heap;

            friend bool parsebc_static(const TokenArray& tarr, ByteCodeModule& mod, size_t& obj);
        public:
            ByteCodeModule(ProgramHeap& heap) : heap(heap) {}
            bool add_block(const std::string& name, const ByteCodeBody& body);
            bool add_static_obj(Obj obj, size_t& addr);                                // add static object to data_segment
            bool add_static_ref(const TokenImp<TokenType::IDN>* label, size_t& addr);  // add static Obj{.ptr} to data_segment
            bool add_static_ref(const std::string& label, const std::string& fname, size_t line_num, size_t col_num, size_t& addr);
            bool export_module(ByteCode& byte_code);
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
            DSP,

            EDG,
            NDE,
            INI,
            BLK,
            NDINP,
            NDOUT,
            BKINP,
            BKOUT,
            PSHMD,
            POPMD,
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

                virtual bool set_labels(const std::map<std::string, size_t>& label_map) override
                {
                    if (!label_map.contains(label))
                        return error::syntax(fname, line_num, col_num, "Unresolved reference to label '%'", label);
                    label_ptr = label_map.at(label);
                    return false;
                }

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *reinterpret_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    *reinterpret_cast<size_t*>(buf) = label_ptr;
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

                virtual bool set_labels(const std::map<std::string, size_t>&) override { return false; }

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *reinterpret_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    *reinterpret_cast<size_t*>(buf) = val;
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

                virtual bool set_labels(const std::map<std::string, size_t>&) override { return false; }

                virtual CodeSegPtr to_bytes(CodeSegPtr buf) const override
                {
                    *reinterpret_cast<InstructionType*>(buf) = OPCODE;
                    buf += sizeof(InstructionType);
                    return buf;
                }
            };

            using enum ::nn::lang::InstructionType;
            using Jmp   = Labeled  < JMP   >;
            using Brt   = Labeled  < BRT   >;
            using Brf   = Labeled  < BRF   >;
            using New   = Valued   < NEW   >;
            using Agg   = Valued   < AGG   >;
            using Arr   = Implicit < ARR   >;
            using Aty   = Valued   < ATY   >;
            using Pop   = Valued   < POP   >;
            using Dup   = Valued   < DUP   >;
            using Cpy   = Implicit < CPY   >;
            using Inst  = Implicit < INST  >;
            using Call  = Implicit < CALL  >;
            using Ret   = Implicit < RET   >;
            using Set   = Implicit < SET   >;
            using IAdd  = Implicit < IADD  >;
            using ISub  = Implicit < ISUB  >;
            using IMul  = Implicit < IMUL  >;
            using IDiv  = Implicit < IDIV  >;
            using IMod  = Implicit < IMOD  >;
            using Add   = Implicit < ADD   >;
            using Sub   = Implicit < SUB   >;
            using Mul   = Implicit < MUL   >;
            using Div   = Implicit < DIV   >;
            using Mod   = Implicit < MOD   >;
            using Eq    = Implicit < EQ    >;
            using Ne    = Implicit < NE    >;
            using Ge    = Implicit < GE    >;
            using Le    = Implicit < LE    >;
            using Gt    = Implicit < GT    >;
            using Lt    = Implicit < LT    >;
            using Idx   = Implicit < IDX   >;
            using XStr  = Implicit < XSTR  >;
            using XFlt  = Implicit < XFLT  >;
            using XInt  = Implicit < XINT  >;
            using Dsp   = Implicit < DSP   >;
            using Edg   = Implicit < EDG   >;
            using Nde   = Implicit < NDE   >;
            using Ini   = Implicit < INI   >;
            using Blk   = Implicit < BLK   >;
            using NdInp = Implicit < NDINP >;
            using NdOut = Implicit < NDOUT >;
            using BkInp = Implicit < BKINP >;
            using BkOut = Implicit < BKOUT >;
            using PshMd = Implicit < PSHMD >;
            using PopMd = Implicit < POPMD >;
            using Ext   = Implicit < EXT   >;
            using Exp   = Implicit < EXP   >;
        }
        
        bool parsebc_static(const TokenArray& tarr, ByteCodeModule& mod, size_t& addr);
        bool parsebc_instruction(const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body);
        bool parsebc_body(const TokenArray& tarr, ByteCodeModule& mod, ByteCodeBody& body);
        bool parsebc_module(const TokenArray& tarr, ByteCodeModule& mod);
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
* Pops the tos, pushes the current pc onto the pc_stack and sets a new pc
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
* dsp          Prints a string at tos to stdout
* 
* Deep learning instructions
* 
* edg          Creates a new edge
* nde          Creates a new named node
* ini          Creates a weight initializer object out of a name and any number of cargs
* blk          Creates a new named block with a given parent
* 
* ndinp        Sets a named node input to an edge
* ndout        Sets a named node output to an edge
* bkinp        Binds a named forward:backward edge pair to a block input
* bkout        Binds a named forward:backward edge pair to a block output
* 
* pshmd        Pushes a new evaluation mode name onto the mode stack
* popmd        Pops the top most evalutation mode name off the mode stack
* 
* ext          Marks a named forward:backward edge pair as a block weight
* exp          Exports a named forward:backward edge pair
* 
*/

#endif
