
void AddSame::eval(RunId id)
{
    void* _0 = inp1->get_data(id);
    void* _1 = inp2->get_data(id);
    void* _2 = out->data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void SubSame::eval(RunId id)
{
    void* _0 = inp1->get_data(id);
    void* _1 = inp2->get_data(id);
    void* _2 = out->data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void MulSame::eval(RunId id)
{
    void* _0 = inp1->get_data(id);
    void* _1 = inp2->get_data(id);
    void* _2 = out->data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void DivSame::eval(RunId id)
{
    void* _0 = inp1->get_data(id);
    void* _1 = inp2->get_data(id);
    void* _2 = out->data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void AddScalar::eval(RunId id)
{
    void* _0 = inp->get_data(id);
    void* _1 = val->get_data(id);
    void* _2 = out->data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void SubScalar::eval(RunId id)
{
    void* _0 = inp->get_data(id);
    void* _1 = val->get_data(id);
    void* _2 = out->data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void MulScalar::eval(RunId id)
{
    void* _0 = inp->get_data(id);
    void* _1 = val->get_data(id);
    void* _2 = out->data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}

void DivScalar::eval(RunId id)
{
    void* _0 = inp->get_data(id);
    void* _1 = val->get_data(id);
    void* _2 = out->data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
        }
    }
    out->id = id;
}
