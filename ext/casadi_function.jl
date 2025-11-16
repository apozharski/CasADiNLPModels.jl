check_arg_type(arg::T1, inarg::T2) where {T1 <: CUSPARSE.CuSparseMatrixCSC{T}, T2 <: SparseMatrixCSC} = true

function check_arg_size(arg::T1, inarg::T2) where {T1 <: CUSPARSE.CuSparseMatrixCSC{T}, T2 <: SparseMatrixCSC}
    len_compat = size(arg) == size(inarg)
    return len_compat &&
           all(rowvals(arg) .== rowvals(inarg)) &&
           all(arg.colPtr .== arg.colptr)
end

