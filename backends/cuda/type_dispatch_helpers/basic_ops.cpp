#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>

using namespace std;

using Cases = vector<pair<string, vector<pair<string, string>>>>;

void write_func_signature(ostream& ofs, const string& cls_name)
{
    ofs << endl << "void " << cls_name << "::eval(RunId id)" << endl << "{" << endl;
}

void write_func_end(ostream& ofs)
{
    ofs << "}" << endl;
}

void write_pre_boilerplate(ostream& ofs, const vector<string>& vars)
{
    int i;
    for (i = 0; i < vars.size(); i++)
        ofs << "    void* _" << i << " = " << vars[i] << "->get_data(id);" << endl;
    ofs << "    void* _" << i << " = out->data;" << endl;
}

void write_post_boilerplate(ostream& ofs)
{
    ofs << "    out->id = id;" << endl;
}

void write_call(ostream& ofs, const string& fn_name, const vector<string>& types)
{
    ofs << fn_name << "<<<(sz + bsz - 1) / bsz, bsz>>>(";
    if (types.size() > 0)
    {
        ofs << "(" << types[0] << "*)_0";
        for (int i = 1; i < types.size(); i++)
            ofs << ", (" << types[i] << "*)_" << i;
    }
    ofs << ", sz);";
}

void write_dispatch(ostream& ofs, const string& indent, const string& fn_name, vector<string>& curr_types, const Cases& cases)
{
    if (cases.size() == 0)
    {
        ofs << indent;
        write_call(ofs, fn_name, curr_types);
        ofs << endl << indent << "break;" << endl;
        return;
    }

    const auto& [switch_var, fw_maps] = cases[0];
    ofs << indent << "switch (" << switch_var << ")" << endl << indent << "{" << endl;
    for (const auto& fw_map : fw_maps)
    {
        ofs << indent << "case " << get<0>(fw_map) << ":\n";
        curr_types.push_back(get<1>(fw_map));
        write_dispatch(ofs, indent + "    ", fn_name, curr_types, vector(cases.begin() + 1, cases.end()));
        curr_types.pop_back();
    }
    ofs << indent << "}" << endl;
}

void write_func(ostream& ofs, const string& cls_name, const string& fn_name, const vector<string>& inp_vars, const Cases& cases)
{
    write_func_signature(ofs, cls_name);
    write_pre_boilerplate(ofs, inp_vars);

    vector<string> curr_types;
    write_dispatch(ofs, "    ", fn_name, curr_types, cases);

    write_post_boilerplate(ofs);
    write_func_end(ofs);
}

int main()
{
    ofstream ofs(SOURCE_DIR"basic_ops_out.cu");

    vector<pair<string, string>> fw_map =
    {
        //{ "core::tensor_dty::F16", "__half" },  Ignoring half precision operations for now...
        { "core::tensor_dty::F32", "float"  },
        { "core::tensor_dty::F64", "double" }
    };

    Cases pointwise_cases =
    {
        { "inp1_dty", fw_map },
        { "inp2_dty", fw_map },
        { "out_dty" , fw_map }
    };

    vector<tuple<string, string, vector<string>, Cases>> pointwise_dispatchers =
    {
        { "AddSame", "add_pointwise", { "inp1", "inp2" }, pointwise_cases },
        { "SubSame", "sub_pointwise", { "inp1", "inp2" }, pointwise_cases },
        { "MulSame", "mul_pointwise", { "inp1", "inp2" }, pointwise_cases },
        { "DivSame", "div_pointwise", { "inp1", "inp2" }, pointwise_cases }
    };

    for (const auto& cfg : pointwise_dispatchers)
        write_func(ofs, get<0>(cfg), get<1>(cfg), get<2>(cfg), get<3>(cfg));

    Cases scalar_cases =
    {
        { "inp_dty", fw_map },
        { "val_dty", fw_map },
        { "out_dty", fw_map }
    };

    vector<tuple<string, string, vector<string>, Cases>> scalar_dispatchers =
    {
        { "AddScalar", "add_scalar", { "inp", "val" }, scalar_cases },
        { "SubScalar", "sub_scalar", { "inp", "val" }, scalar_cases },
        { "MulScalar", "mul_scalar", { "inp", "val" }, scalar_cases },
        { "DivScalar", "div_scalar", { "inp", "val" }, scalar_cases }
    };

    for (const auto& cfg : scalar_dispatchers)
        write_func(ofs, get<0>(cfg), get<1>(cfg), get<2>(cfg), get<3>(cfg));

    ofs.close();
}
