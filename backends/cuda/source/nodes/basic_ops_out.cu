
void AddSame::forward(RunId id)
{
    inp1->forward(id);
    inp2->forward(id);
    void* _0 = inp1->forward_data;
    void* _1 = inp2->forward_data;
    void* _2 = out->forward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void SubSame::forward(RunId id)
{
    inp1->forward(id);
    inp2->forward(id);
    void* _0 = inp1->forward_data;
    void* _1 = inp2->forward_data;
    void* _2 = out->forward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void MulSame::forward(RunId id)
{
    inp1->forward(id);
    inp2->forward(id);
    void* _0 = inp1->forward_data;
    void* _1 = inp2->forward_data;
    void* _2 = out->forward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void DivSame::forward(RunId id)
{
    inp1->forward(id);
    inp2->forward(id);
    void* _0 = inp1->forward_data;
    void* _1 = inp2->forward_data;
    void* _2 = out->forward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void AddScalar::forward(RunId id)
{
    inp->forward(id);
    val->forward(id);
    void* _0 = inp->forward_data;
    void* _1 = val->forward_data;
    void* _2 = out->forward_data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void SubScalar::forward(RunId id)
{
    inp->forward(id);
    val->forward(id);
    void* _0 = inp->forward_data;
    void* _1 = val->forward_data;
    void* _2 = out->forward_data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void MulScalar::forward(RunId id)
{
    inp->forward(id);
    val->forward(id);
    void* _0 = inp->forward_data;
    void* _1 = val->forward_data;
    void* _2 = out->forward_data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void DivScalar::forward(RunId id)
{
    inp->forward(id);
    val->forward(id);
    void* _0 = inp->forward_data;
    void* _1 = val->forward_data;
    void* _2 = out->forward_data;
    switch (inp_dty)
    {
    case core::tensor_dty::F32:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (val_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_scalar_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    out->forward_id = id;
}

void AddSame::backward(RunId id)
{
    inp1->backward(id);
    inp2->backward(id);
    void* _0 = inp1->backward_data;
    void* _1 = inp2->backward_data;
    void* _2 = out->backward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                add_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    inp1->backward_id = id;
    inp2->backward_id = id;
}

void SubSame::backward(RunId id)
{
    inp1->backward(id);
    inp2->backward(id);
    void* _0 = inp1->backward_data;
    void* _1 = inp2->backward_data;
    void* _2 = out->backward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                sub_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    inp1->backward_id = id;
    inp2->backward_id = id;
}

void MulSame::backward(RunId id)
{
    inp1->backward(id);
    inp2->backward(id);
    void* _0 = inp1->backward_data;
    void* _1 = inp2->backward_data;
    void* _2 = out->backward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                mul_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    inp1->backward_id = id;
    inp2->backward_id = id;
}

void DivSame::backward(RunId id)
{
    inp1->backward(id);
    inp2->backward(id);
    void* _0 = inp1->backward_data;
    void* _1 = inp2->backward_data;
    void* _2 = out->backward_data;
    switch (inp1_dty)
    {
    case core::tensor_dty::F32:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    case core::tensor_dty::F64:
        switch (inp2_dty)
        {
        case core::tensor_dty::F32:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (float*)_1, (double*)_2, sz);
                break;
            }
            break;
        case core::tensor_dty::F64:
            switch (out_dty)
            {
            case core::tensor_dty::F32:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (float*)_2, sz);
                break;
            case core::tensor_dty::F64:
                div_pointwise_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)_0, (double*)_1, (double*)_2, sz);
                break;
            }
            break;
        }
        break;
    }
    inp1->backward_id = id;
    inp2->backward_id = id;
}
