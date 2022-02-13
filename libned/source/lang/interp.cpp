#include <ned/errors.h>
#include <ned/lang/interp.h>
#include <ned/lang/bytecode.h>

#include <iostream>
#include <vector>

#define oprand (*((size_t*)(code + pc)))

namespace nn
{
    namespace lang
    {
        // state of the interpreter (no, this shouldn't be implemented as a singleton.)
        // programming state
        size_t pc = 0;
        CodeSegPtr code;
        DataSegPtr data;
        bool complete;
        std::vector<size_t> pc_stack;

        // deep learning state
        GraphBuilder* pbuilder;
        std::vector<std::string> md_stack;

        // graph builder operations

        GraphBuilder::GraphBuilder() {}

        GraphBuilder::~GraphBuilder()
        {
            if (is_exported)
            {
                // Figure out how exporting works
            }
            else
            {
                // delete everything
            }
        }

        inline bool GraphBuilder::edge_exists(uint64_t edge)
        {
            return edge * (edge < edges.size());
        }

        inline bool GraphBuilder::node_exists(uint64_t node)
        {
            return node * (node < nodes.size());
        }

        inline bool GraphBuilder::init_exists(uint64_t init)
        {
            return init * (init < inits.size());
        }

        inline bool GraphBuilder::tensor_exists(uint64_t tensor)
        {
            return tensor * (tensor < tensors.size());
        }

        inline bool GraphBuilder::block_exists(uint64_t block)
        {
            return block * (block < blocks.size());
        }

        std::string GraphBuilder::current_mode()
        {
            if (md_stack.size() == 0)
                return "";

            // Not very efficient, but it'll work
            std::stringstream ss;
            ss << md_stack[0];
            for (size_t i = 1; i < md_stack.size(); i++)
                ss << "." << md_stack[i];
            return ss.str();
        }

        bool GraphBuilder::create_edg(Obj& obj, const core::EdgeInfo& info)
        {
            obj.ptr = edges.size();
            edges.push_back(new EdgeBuilder{ .info = info });
            return false;
        }

        bool GraphBuilder::create_tsr(Obj& obj)
        {
            obj.ptr = tensors.size();
            tensors.push_back(new TensorBuilder());
            return false;
        }

        bool GraphBuilder::create_nde(Obj& obj, const std::string& name)
        {
            obj.ptr = nodes.size();
            nodes.push_back(new NodeBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::create_ini(Obj& obj, const std::string& name)
        {
            obj.ptr = inits.size();
            inits.push_back(new InitBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::create_blk(Obj& obj, const std::string& name)
        {
            obj.ptr = blocks.size();
            blocks.push_back(new BlockBuilder{ .name = name });
            return false;
        }

        bool GraphBuilder::get_fwd(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->fwd_edge;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized forward edge of a tensor");
            return false;
        }

        bool GraphBuilder::get_bwd(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->bwd_edge;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized backward edge of a tensor");
            return false;
        }

        bool GraphBuilder::get_ini(Obj& obj, uint64_t tensor)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            obj.ptr = tensors[tensor]->init;
            if (!obj.ptr)
                return error::runtime("Attempted to retrieve the uninitialized weight initializer of a tensor");
            return false;
        }

        bool GraphBuilder::set_fwd(uint64_t tensor, uint64_t edge)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            tensors[tensor]->fwd_edge = edge;
            return false;
        }

        bool GraphBuilder::set_bwd(uint64_t tensor, uint64_t edge)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            tensors[tensor]->bwd_edge = edge;
            return false;
        }

        bool GraphBuilder::set_ini(uint64_t tensor, uint64_t init)
        {
            assert(!is_exported);

            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");
            if (!init_exists(init))
                return error::runtime("Attempted to reference a non-existant weight initializer");
            tensors[tensor]->init = init;
            return false;

        }

        bool GraphBuilder::mrg(uint64_t lhs_edge, uint64_t rhs_edge)
        {
            assert(!is_exported);

            if (!edge_exists(lhs_edge) || !edge_exists(rhs_edge))
                return error::runtime("Attempted to reference a non-existant edge object");

            // Merging the edge inputs
            for (const auto& [md, conn] : edges[rhs_edge]->md_inps)
            {
                if (edges[lhs_edge]->md_inps.contains(md))
                    return error::runtime("Attempted to merge two edges when both had bound inputs");
                assert(nodes[conn.node]->outs.contains(md));
                nodes[conn.node]->outs[md] = lhs_edge;
                edges[lhs_edge]->md_inps[md] = conn;
            }
            edges[rhs_edge]->md_inps.clear();

            // Merging the edge outputs
            for (const auto& [md, conns] : edges[rhs_edge]->md_outs)
                for (const auto& conn : conns)
                {
                    nodes[conn.node]->inps[conn.name] = lhs_edge;
                    edges[lhs_edge]->md_outs[md].push_back(conn);  // dupicates will be handled during the export
                }

            delete edges[rhs_edge];
            edges[rhs_edge] = edges[lhs_edge];
            return false;
        }

        bool GraphBuilder::add_ndcfg(const std::string& name, uint64_t node, core::Config* pconfig)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (nodes[node]->configs.contains(name))
                return error::runtime("Attempted to overwrite node configuration '%'", name);
            nodes[node]->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::add_bkcfg(const std::string& name, uint64_t block, core::Config* pconfig)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (blocks[block]->configs.contains(name))
                return error::runtime("Attempted to overwrite block configuration '%'", name);
            blocks[block]->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::add_incfg(const std::string& name, uint64_t init, core::Config* pconfig)
        {
            assert(!is_exported);

            if (!init_exists(init))
                return error::runtime("Attempted to reference a non-existant weight initializer");
            if (inits[init]->configs.contains(name))
                return error::runtime("Attempted to overwrite init configuration '%'", name);
            inits[init]->configs[name] = std::unique_ptr<core::Config>(pconfig);
            return false;
        }

        bool GraphBuilder::set_ndinp(const std::string& name, uint64_t node, uint64_t edge)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            if (nodes[node]->inps.contains(name))
                return error::runtime("Attempted to bind node input '%' multiple times", name);
            nodes[node]->inps[name] = edge;
            edges[edge]->md_outs[current_mode()].push_back({ node, name });
            return false;
        }

        bool GraphBuilder::set_ndout(const std::string& name, uint64_t node, uint64_t edge)
        {
            assert(!is_exported);

            if (!node_exists(node))
                return error::runtime("Attempted to reference a non-existant node");
            if (!edge_exists(edge))
                return error::runtime("Attempted to reference a non-existant edge");
            if (nodes[node]->outs.contains(name))
                return error::runtime("Attempted to bind node output '%' multiple times", name);
            std::string md = current_mode();
            if (edges[edge]->md_inps.contains(md))
                return error::runtime("Attempted to bind an edge input multiple times");
            nodes[node]->outs[name] = edge;
            edges[edge]->md_inps[md] = { node, name };
            return false;
        }

        bool GraphBuilder::set_bkprt(uint64_t block, uint64_t parent)
        {
            assert(!is_exported);

            if (!block_exists(block) || parent >= blocks.size())  // parent can be null
                return error::runtime("Attempted to reference a non-existant block");

            if (blocks[block]->parent)
                return error::runtime("Attempted to set a block's parent when it has already been set");
            else if (block == root)
                return error::runtime("Attempted to set the root block's parent");

            blocks[block]->parent = parent;
            if (parent)
                return false;

            if (root)
                return error::runtime("Attempted to set a block as the root, when a root was already set");
            root = block;
            return false;
        }

        bool GraphBuilder::set_bkinp(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->inps.contains(name))
                return error::runtime("Attempted to bind block input '%' multiple times", name);
            if (!tensors[tensor]->fwd_edge)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized forward edge", name);
            if (!tensors[tensor]->bwd_edge)
                return error::runtime("Attempted to bind block input '%' to a tensor with an uninitialized backward edge", name);
            blocks[block]->inps[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkout(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->outs.contains(name))
                return error::runtime("Attempted to bind block output '%' multiple times", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized forward edge", name);
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to bind block output '%' to a tensor with an uninitialized backward edge", name);
            blocks[block]->outs[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkext(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->exts.contains(name))
                return error::runtime("Attempted to overwrite block weight '%'", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to extern a tensor with an uninitialized forward edge");
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to extern a tensor with an uninitialized backward edge");
            blocks[block]->exts[name] = tensor;
            return false;
        }

        bool GraphBuilder::set_bkexp(const std::string& name, uint64_t block, uint64_t tensor)
        {
            assert(!is_exported);

            if (!block_exists(block))
                return error::runtime("Attempted to reference a non-existant block");
            if (!tensor_exists(tensor))
                return error::runtime("Attempted to reference a non-existant tensor");

            if (blocks[block]->exps.contains(name))
                return error::runtime("Attempted to overwrite model export '%'", name);
            if (tensors[tensor]->fwd_edge == 0)
                return error::runtime("Attempted to export a tensor with an uninitialized forward edge");
            if (tensors[tensor]->bwd_edge == 0)
                return error::runtime("Attempted to export a tensor with an uninitialized backward edge");
            blocks[block]->exps[name] = tensor;
            return false;
        }

        bool GraphBuilder::export_graph(core::Graph& graph)
        {
            assert(!is_exported);

            // TODO: implement graph exporting
            is_exported = true;
            return error::runtime("Graph exporting has not been implemented yet...");
        }

        // stack operations

        bool CallStack::pop(Obj& obj)
        {
            if (sp == 0)
                return error::runtime("Stack pointer out of bounds during pop operation");
            obj = stack[--sp];
            return false;
        }

        bool CallStack::del(size_t i)
        {
            if (i >= sp)
                return error::runtime("Attempted to delete a non-existent stack element");
            sp--;
            for (size_t j = sp - i; j < sp; j++)
                stack[j] = stack[j + 1];
            return false;
        }

        bool CallStack::get(size_t i, Obj& obj)
        {
            if (i >= sp)
                return error::runtime("Attempted to retrieve a non-existent stack element");
            obj = stack[sp - i - 1];
            return false;
        }

        bool CallStack::push(Obj obj)
        {
            if (sp >= stack.size())
                return error::runtime("Stack overflow error");
            stack[sp++] = obj;
            return false;
        }

        // helper functions

        inline bool set_pc(size_t val)
        {
            pc = val;
            return false;
        }

        inline bool push_pc()
        {
            pc_stack.push_back(pc);
            return false;
        }

        inline bool pop_pc()
        {
            assert(pc_stack.size() > 0);
            pc = pc_stack.back();
            pc_stack.pop_back();
            return false;
        }

        inline bool push_md(const std::string& mode)
        {
            md_stack.push_back(mode);
            return false;
        }

        inline bool pop_md()
        {
            if (md_stack.size() == 0)
                return error::runtime("Attempted to release a non-existent evaluation mode");
            md_stack.pop_back();
            return false;
        }

        // instruction implementations

        inline bool exec_jmp()
        {
            return
                set_pc(oprand);
        }

        inline bool exec_brt(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? oprand : pc + sizeof(size_t));
        }

        inline bool exec_brf(CallStack& stack)
        {
            Obj obj;
            return
                stack.pop(obj) ||
                set_pc(*obj.bool_obj ? pc + sizeof(size_t) : oprand);
        }

        inline bool exec_new(CallStack& stack)
        {
            return
                stack.push(data[oprand]) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_agg(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<Obj> objs(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj obj;
                if (stack.pop(obj))
                    return true;
                objs.push_back(obj);
            }
            Obj dst;
            return
                heap.create_obj_agg(dst, objs) ||
                stack.push(dst) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_arr(CallStack& stack, ProgramHeap& heap)
        {
            Obj elem_ty, arr_ty;
            return
                stack.pop(elem_ty) ||
                heap.create_type_arr(arr_ty, elem_ty.type_obj) ||
                stack.push(arr_ty);
        }

        inline bool exec_aty(CallStack& stack, ProgramHeap& heap)
        {
            std::vector<TypeObj*> tys(oprand);
            for (size_t i = 0; i < oprand; i++)
            {
                Obj ty;
                if (stack.pop(ty))
                    return true;
                tys.push_back(ty.type_obj);
            }
            Obj type;
            return
                heap.create_type_agg(type, tys) ||
                stack.push(type) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_pop(CallStack& stack)
        {
            return
                stack.del(oprand) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_dup(CallStack& stack)
        {
            Obj obj;
            return
                stack.get(oprand, obj) ||
                stack.push(obj) ||
                set_pc(pc + sizeof(size_t));
        }

        inline bool exec_cpy(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->cpy(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_inst(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, inst;
            return
                stack.pop(type) ||
                type.type_obj->inst(heap, inst) ||
                stack.push(inst);
        }

        inline bool exec_call(CallStack& stack)
        {
            Obj proc;
            return
                stack.pop(proc) ||
                push_pc() ||
                set_pc(proc.ptr);
        }

        inline bool exec_ret(CallStack& stack)
        {
            if (pc_stack.size() > 0)
                return pop_pc();

            complete = true;
            return false;
        }

        inline bool exec_set(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->set(heap, lhs, rhs);
        }

        inline bool exec_iadd(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->iadd(heap, lhs, rhs);
        }

        inline bool exec_isub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->isub(heap, lhs, rhs);
        }

        inline bool exec_imul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imul(heap, lhs, rhs);
        }

        inline bool exec_idiv(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idiv(heap, lhs, rhs);
        }

        inline bool exec_imod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->imod(heap, lhs, rhs);
        }

        inline bool exec_add(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->add(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_sub(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->sub(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mul(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mul(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_div(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->div(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_mod(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->mod(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_eq(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->eq(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ne(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ne(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_gt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->gt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_lt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->lt(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_ge(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->ge(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_le(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->le(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_idx(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, rhs, lhs, dst;
            return
                stack.pop(type) ||
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                type.type_obj->idx(heap, dst, lhs, rhs) ||
                stack.push(dst);
        }

        inline bool exec_xstr(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xstr(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xflt(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xflt(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_xint(CallStack& stack, ProgramHeap& heap)
        {
            Obj type, src, dst;
            return
                stack.pop(type) ||
                stack.pop(src) ||
                type.type_obj->xint(heap, dst, src) ||
                stack.push(dst);
        }

        inline bool exec_dsp(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            std::cout << *obj.str_obj << std::endl;
            return false;
        }

        inline bool exec_err(CallStack& stack)
        {
            Obj obj;
            if (stack.pop(obj))
                return true;
            return error::runtime(*obj.str_obj);
        }

        inline bool exec_edg(CallStack& stack)
        {
            Obj argnum;
            if (stack.pop(argnum))
                return true;
            std::vector<size_t> dims;
            dims.resize(*argnum.int_obj);
            for (IntObj i = 0; i < *argnum.int_obj; i++)
            {
                Obj dim;
                if (stack.pop(dim))
                    return true;
                dims.push_back(*dim.int_obj);
            }
            Obj obj, fty;
            return
                stack.pop(fty) ||
                pbuilder->create_edg(obj, core::EdgeInfo{ *fty.fty_obj, dims }) ||
                stack.push(obj);
        }

        inline bool exec_tsr(CallStack& stack)
        {
            Obj obj;
            return
                pbuilder->create_tsr(obj) ||
                stack.push(obj);
        }

        inline bool exec_nde(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_nde(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_ini(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_ini(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_blk(CallStack& stack)
        {
            Obj obj, name;
            return
                stack.pop(name) ||
                pbuilder->create_blk(obj, *name.str_obj) ||
                stack.push(obj);
        }

        inline bool exec_gfwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_fwd(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_gbwd(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_bwd(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_gini(CallStack& stack)
        {
            Obj obj, ten;
            return
                stack.pop(ten) ||
                pbuilder->get_ini(obj, ten.ptr) ||
                stack.push(obj);
        }

        inline bool exec_sfwd(CallStack& stack)
        {
            Obj tensor, edge;
            return
                stack.pop(edge) ||
                stack.pop(tensor) ||
                pbuilder->set_fwd(tensor.ptr, edge.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_sbwd(CallStack& stack)
        {
            Obj tensor, edge;
            return
                stack.pop(edge) ||
                stack.pop(tensor) ||
                pbuilder->set_bwd(tensor.ptr, edge.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_sini(CallStack& stack)
        {
            Obj tensor, init;
            return
                stack.pop(init) ||
                stack.pop(tensor) ||
                pbuilder->set_ini(tensor.ptr, init.ptr) ||
                stack.push(tensor);
        }

        inline bool exec_mrg(CallStack& stack)
        {
            Obj lhs, rhs;
            return
                stack.pop(rhs) ||
                stack.pop(lhs) ||
                pbuilder->mrg(lhs.ptr, rhs.ptr);
        }

        inline bool exec_ndcfg(CallStack& stack)
        {
            Obj node, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                stack.pop(node) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_ndcfg(*name.str_obj, node.ptr, cfg) ||
                stack.push(node);
        }

        inline bool exec_bkcfg(CallStack& stack)
        {
            Obj block, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                stack.pop(block) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_bkcfg(*name.str_obj, block.ptr, cfg) ||
                stack.push(block);
        }

        inline bool exec_incfg(CallStack& stack)
        {
            Obj init, name, type, obj;
            core::Config* cfg;
            return
                stack.pop(name) ||
                stack.pop(type) ||
                stack.pop(obj) ||
                stack.pop(init) ||
                type.type_obj->cfg(cfg, obj) ||
                pbuilder->add_incfg(*name.str_obj, init.ptr, cfg) ||
                stack.push(init);
        }

        inline bool exec_ndinp(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(name) ||
                stack.pop(edge) ||
                stack.pop(node) ||
                pbuilder->set_ndinp(*name.str_obj, node.ptr, edge.ptr) ||
                stack.push(node);
        }

        inline bool exec_ndout(CallStack& stack)
        {
            Obj name, node, edge;
            return
                stack.pop(name) ||
                stack.pop(edge) ||
                stack.pop(node) ||
                pbuilder->set_ndout(*name.str_obj, node.ptr, edge.ptr) ||
                stack.push(node);
        }

        inline bool exec_bkprt(CallStack& stack)
        {
            Obj block, parent;
            return
                stack.pop(block) ||
                stack.pop(parent) ||
                pbuilder->set_bkprt(block.ptr, parent.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkinp(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkinp(*name.str_obj, block.ptr, tensor.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkout(CallStack& stack)
        {
            Obj name, block, tensor;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkout(*name.str_obj, block.ptr, tensor.ptr) ||
                stack.push(block);
        }

        inline bool exec_bkext(CallStack& stack)
        {
            Obj name, tensor, block;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkext(*name.str_obj, block.ptr, tensor.ptr);
        }

        inline bool exec_bkexp(CallStack& stack)
        {
            Obj name, tensor, block;
            return
                stack.pop(name) ||
                stack.pop(tensor) ||
                stack.pop(block) ||
                pbuilder->set_bkexp(*name.str_obj, block.ptr, tensor.ptr);
        }

        inline bool exec_pshmd(CallStack& stack)
        {
            Obj mode;
            return
                stack.pop(mode) ||
                push_md(*mode.str_obj);
        }

        inline bool exec_popmd(CallStack& stack)
        {
            return
                pop_md();
        }

        bool exec(CallStack& stack, ProgramHeap& heap, GraphBuilder& builder, ByteCode& byte_code, std::string entry_point)
		{
            if (!byte_code.proc_offsets.contains(entry_point))
                return error::runtime("Unable to find entry point '%'", entry_point);

            // Initializing the interpreter state
            pc = byte_code.proc_offsets.at(entry_point);
            code = byte_code.code_segment;
            data = byte_code.data_segment;
            complete = false;
            pc_stack.clear();
            pbuilder = &builder;
            md_stack.clear();
            error::bind_runtime_context(byte_code.debug_info, pc);
            if (stack.push(Obj{ .ptr = 0 }))  // this will mark the first block as root
                return true;

            InstructionType ty;
            while (!complete)
            {
                ty = *(InstructionType*)(code + pc);
                pc += sizeof(InstructionType);
                switch (ty)
                {
                case InstructionType::JMP:
                    if (exec_jmp())
                        goto runtime_error;
                    break;
                case InstructionType::BRT:
                    if (exec_brt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BRF:
                    if (exec_brf(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NEW:
                    if (exec_new(stack))
                        goto runtime_error;
                    break;
                case InstructionType::AGG:
                    if (exec_agg(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ARR:
                    if (exec_arr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ATY:
                    if (exec_aty(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::POP:
                    if (exec_pop(stack))
                        goto runtime_error;
                    break;
                case InstructionType::DUP:
                    if (exec_dup(stack))
                        goto runtime_error;
                    break;
                case InstructionType::CPY:
                    if (exec_cpy(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::INST:
                    if (exec_inst(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::CALL:
                    if (exec_call(stack))
                        goto runtime_error;
                    break;
                case InstructionType::RET:
                    if (exec_ret(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SET:
                    if (exec_set(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IADD:
                    if (exec_iadd(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ISUB:
                    if (exec_isub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMUL:
                    if (exec_imul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDIV:
                    if (exec_idiv(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IMOD:
                    if (exec_imod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::ADD:
                    if (exec_add(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::SUB:
                    if (exec_sub(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MUL:
                    if (exec_mul(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DIV:
                    if (exec_div(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::MOD:
                    if (exec_mod(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::EQ:
                    if (exec_eq(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::NE:
                    if (exec_ne(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GT:
                    if (exec_gt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LT:
                    if (exec_lt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::GE:
                    if (exec_ge(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::LE:
                    if (exec_le(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::IDX:
                    if (exec_idx(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XSTR:
                    if (exec_xstr(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XFLT:
                    if (exec_xflt(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::XINT:
                    if (exec_xint(stack, heap))
                        goto runtime_error;
                    break;
                case InstructionType::DSP:
                    if (exec_dsp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::ERR:
                    if (exec_err(stack))
                        goto runtime_error;
                    break;

                case InstructionType::EDG:
                    if (exec_edg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::TSR:
                    if (exec_tsr(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDE:
                    if (exec_nde(stack))
                        goto runtime_error;
                    break;
                case InstructionType::INI:
                    if (exec_ini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BLK:
                    if (exec_blk(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GFWD:
                    if (exec_gfwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GBWD:
                    if (exec_gbwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::GINI:
                    if (exec_gini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SFWD:
                    if (exec_sfwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SBWD:
                    if (exec_sbwd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::SINI:
                    if (exec_sini(stack))
                        goto runtime_error;
                    break;
                case InstructionType::MRG:
                    if (exec_mrg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDCFG:
                    if (exec_ndcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKCFG:
                    if (exec_bkcfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::INCFG:
                    if (exec_incfg(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDINP:
                    if (exec_ndinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::NDOUT:
                    if (exec_ndout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKPRT:
                    if (exec_bkprt(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKINP:
                    if (exec_bkinp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKOUT:
                    if (exec_bkout(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKEXT:
                    if (exec_bkext(stack))
                        goto runtime_error;
                    break;
                case InstructionType::BKEXP:
                    if (exec_bkexp(stack))
                        goto runtime_error;
                    break;
                case InstructionType::PSHMD:
                    if (exec_pshmd(stack))
                        goto runtime_error;
                    break;
                case InstructionType::POPMD:
                    if (exec_popmd(stack))
                        goto runtime_error;
                    break;

                default:
                    error::runtime("Invalid instruction opcode '%'", (uint8_t)ty);
                    goto runtime_error;
                }
            }
            return false;

        runtime_error:
            // TODO: unwind the call stack with location info
            return true;
		}
	}
}
