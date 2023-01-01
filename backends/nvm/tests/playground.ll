@global_var = internal global [4 x float] zeroinitializer

define i1 @nvm_streq(i8* %lhs, i8* %rhs) {
entry:
    br label %loop

loop:
    %idx = phi i32 [0, %entry], [%next_idx, %cont]
    %lchptr = getelementptr i8, i8* %lhs, i32 %idx
    %rchptr = getelementptr i8, i8* %rhs, i32 %idx
    %lch = load i8, i8* %lchptr
    %rch = load i8, i8* %rchptr
    %cheq = icmp ne i8 %lch, %rch
    br i1 %cheq, label %ret_false, label %eq

eq:
    %chz = icmp eq i8 %lch, 0
    br i1 %chz, label %ret_true, label %cont

cont:
    %next_idx = add i32 %idx, 1
    br label %loop

ret_true:
    ret i1 1

ret_false:
    ret i1 0
}

define internal void @nvm_memcpy(i8* %dst, i8* %src, i32 %nbytes) {
entry:
    br label %loop

loop:
    %idx = phi i32 [0, %entry], [%nidx, %loop]
    %pdst = getelementptr i8, i8* %dst, i32 %idx
    %psrc = getelementptr i8, i8* %src, i32 %idx
    %tmp = load i8, i8* %psrc
    store i8 %tmp, i8* %pdst
    %nidx = add i32 %idx, 1
    %cond = icmp eq i32 %nidx, %nbytes
    br i1 %cond, label %end, label %loop

end:
    ret void
}

define dllexport void @setter(i8* %src) {
entry:
    %cast = bitcast [4 x float]* @global_var to i8*
    call void @nvm_memcpy(i8* %cast, i8* %src, i32 16)
    ret void
}

define dllexport void @getter(i8* %dst) {
entry:
    %cast = bitcast [4 x float]* @global_var to i8*
    call void @nvm_memcpy(i8* %dst, i8* %cast, i32 16)
    ret void
}

@_fltused = constant i32 0

define dllexport void @matmulk0(float* %out) {
entry:
    br label %loop

loop:
    %idx = phi i32 [0, %entry], [%nidx, %loop]
    %ptr = getelementptr float, float* %out, i32 %idx
    store float 0.0, ptr %ptr
    %nidx = add i32 %idx, 1
    %cond = icmp eq i32 %nidx, 4
    br i1 %cond, label %end, label %loop

end:
    ret void
}
