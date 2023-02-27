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

@lhs = internal global [4 x float] zeroinitializer
@rhs = internal global [4 x float] zeroinitializer
@out = internal global [4 x float] zeroinitializer

define void @__add__() {
entry:
    %lhs_elem = get float, @lhs, []
    %rhs_elem = get float, @rhs, []
    %out_elem = add float, %lhs_elem, %rhs_elem
    set float, @out, [], %out_elem
    ret void
}

define void @__add__() {
entry:
    br label %start_loop0
start_loop0:
    %idx0 = phi i32 [0, %entry], [%nidx0, %end_loop0]
    %cond0 = icmp eq i32 %idx0, 10
    br i1 %cond0, label %end, label %body
body:
    %lhs_elem = get float, @lhs, [%idx0]
    %rhs_elem = get float, @rhs, [%idx0]
    %out_elem = add float, %lhs_elem, %rhs_elem
    set float, @out, [%idx0], %out_elem
    br label %end_loop0
end_loop0:
    %nidx0 = add i32 %idx0, 1
    br label %start_loop0
end:
    ret void
}

define void @__add__() {
entry:
    br label %start_loop0
start_loop0:
    %idx0 = phi i32 [0, %entry], [%nidx0, %end_loop0]
    %cond0 = icmp eq i32 %idx0, 10
    br i1 %cond0, label %end, label %start_loop1
start_loop1:
    %idx1 = phi i32 [0, %start_loop0], [%nidx1, %end_loop1]
    %cond1 = icmp eq i32 %idx1, 10
    br i1 %cond1, label %end_loop1, label %body
body:
    %lhs_elem = get float, @lhs, [%idx0, %idx1]
    %rhs_elem = get float, @rhs, [%idx0, %idx1]
    %out_elem = add float, %lhs_elem, %rhs_elem
    set float, @out, [%idx0, %idx1], %out_elem
    br label %end_loop1
end_loop1:
    %nidx1 = add i32 %idx1, 1
    br label %start_loop1
end_loop0:
    %nidx0 = add i32 %idx0, 1
    br label %start_loop0
end:
    ret void
}
