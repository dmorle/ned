#ifndef NED_BYTECODE_H
#define NED_BYTECODE_H

#include <string>
#include <vector>
#include <memory>
#include <map>

#include <ned/lang/obj.h>

namespace nn
{
    namespace lang
    {
        class Instruction
        {
        public:
            virtual size_t size() const = 0;
            virtual void* to_bytes(void* buf) const = 0;

            virtual void set_labels(const std::map<std::string, size_t>& label_map) = 0;
        };

        class StaticRef {};

        class StaticsBuilder
        {
        public:
            StaticRef get(std::string ty, std::string val);
        };

        class CodeBlock
        {
            std::string name;
            std::vector<std::unique_ptr<Instruction>> instructions;
            std::map<std::string, size_t> var_map;
            std::map<std::string, size_t> label_map;

            template<typename T> static int get_stack_change();

        public:
            size_t size() const;
            void* to_bytes(void* buf, size_t block_offset) const;
            void set_var_name(const std::string& var);
            void add_label(const std::string& var);

            template<typename T> void add_instruction(const T& inst)
            {
                constexpr int n = CodeBlock::get_stack_change<T>();
                std::vector<const std::string&> popped_vars;
                if (n != 0)  // Hopefully this will get optimized away when the template gets expanded
                    for (auto& [key, val] : var_map)
                        val += n;
                if (n < 0)  // Hopefully this will also get optimized away when the template gets expanded
                    for (const auto& e : popped_vars)
                        var_map.erase(e);
                instructions.push_back(std::make_unique(inst));
            }
        };

        enum class InstructionTypes : uint8_t
        {
            JMP,
            BRT,
            BRF,
            POP,
            NEW,
            ARR,
            TUP,
            TEN,
            INST,
            TYPE,
            DUP,
            CARG,
            CALL,
            RET,
            SET,
            ADD,
            SUB,
            MUL,
            DIV,
            MOD,
            IDX,
            EQ,
            NE,
            GT,
            LT,
            GE,
            LE
        };

        namespace instruction
        {
            class Labeled :
                public Instruction
            {
                std::string label;
                size_t label_idx = 0;
            public:
                Labeled(std::string label) : label(label) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Jmp :
                public Labeled
            {
            public:
                using Labeled::Labeled;

                virtual void* to_bytes(void* buf) const;
            };

            class Brt :
                public Labeled
            {
            public:
                using Labeled::Labeled;

                virtual void* to_bytes(void* buf) const;
            };

            class Brf :
                public Labeled
            {
            public:
                using Labeled::Labeled;

                virtual void* to_bytes(void* buf) const;
            };

            class Pop :
                public Instruction
            {
                size_t n;
            public:
                Pop(size_t n) : n(n) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class New :
                public Instruction
            {
                StaticRef ref;
            public:
                New(StaticRef ref);

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Arr :
                public Instruction
            {
                size_t n;
            public:
                Arr(size_t n) : n(n) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Tup :
                public Instruction
            {
                size_t n;
            public:
                Tup(size_t n) : n(n) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Inst :
                public Instruction
            {
            public:
                Inst() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Type :
                public Instruction
            {
            public:
                Type() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Dup :
                public Instruction
            {
                size_t n;
            public:
                Dup(size_t n) : n(n) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Carg :
                public Instruction
            {
                size_t n;
            public:
                Carg(size_t n) : n(n) {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Call :
                public Instruction
            {
            public:
                Call() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Ret :
                public Instruction
            {
            public:
                Ret() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Add :
                public Instruction
            {
            public:
                Add() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Sub :
                public Instruction
            {
            public:
                Sub() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Mul :
                public Instruction
            {
            public:
                Mul() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Div :
                public Instruction
            {
            public:
                Div() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Mod :
                public Instruction
            {
            public:
                Mod() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Eq :
                public Instruction
            {
            public:
                Eq() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Ne :
                public Instruction
            {
            public:
                Ne() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Ge :
                public Instruction
            {
            public:
                Ge() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Le :
                public Instruction
            {
            public:
                Le() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Gt :
                public Instruction
            {
            public:
                Gt() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Lt :
                public Instruction
            {
            public:
                Lt() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };

            class Idx :
                public Instruction
            {
            public:
                Idx() {}

                virtual size_t size() const;
                virtual void* to_bytes(void* buf) const;
                virtual void set_labels(const std::map<std::string, size_t>& label_map);
            };
        }
    }
}

/*
jmp <label>  Unconditional Jump
brt <label>  Branch if true with pop
brf <label>  Branch if false with pop
pop <int>    Pops element <int> off the stack

new <addr>   Adds an object from static memory onto the stack
arr <int>    Creates a new array from <int> elements on the stack
tup <int>    Creates a new tuple from <int> elements on the stack

inst         Creates an instance from the element on the top of the stack, and pops the type
type         Creates a type from the element on the top of the stack, and pops the inst
dup <int>    Duplicates an object on the stack

Applies the top <int> elements on the stack as cargs to the next object on the stack
carg <int>

Applies the top <int> elements on the stack as vargs to the next object on the stack
Relies on the object to push the cargs and vargs onto the stack
An element is pushed onto the call stack which contains the program counter and a reference to the caller
call

Retrieves the program counter and a reference to the caller from the call stack
Relies on the caller to cleanup the stack from the call, and updates the program counter
ret

Binary Operations; Pops the top two elements off the stack, does the op, pushes the result

add  Adds two objects
sub  Subtracts two objects
mul  Multiplies two objects
div  Divides two objects
mod  Calculates the modulus between two objects

eq   Checks if two objects are equal
ne   Checks if two objects are not equal
ge   Checks if the left is greater than or equal to the right
le   Checks if the left if less than or equal to the right
gt   Checks if the left is greater than the right
lt   Checks if the left is less than the right

idx  Retrieves the left at the index of the right

*/

#endif
