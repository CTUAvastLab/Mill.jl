using CuArrays
using SparseArrays

"""
    Generic GPU conversion utility for structs
"""
@generated function gpu(x)
    expr = :($(get_unionall(x))())
    for field in fieldnames(x)
        if field == :metadata
            push!(expr.args, :(x.$field))
        else
            push!(expr.args, :(gpu(x.$field)))
        end
    end

    expr
end

SparseArrays.SparseMatrixCSC(oh::Flux.OneHotMatrix) =
    SparseMatrixCSC(size(oh,1), size(oh,2), collect(UInt32(1):UInt32(1+size(oh,2))), map(ohv->ohv.ix, oh.data), ones(Float32, size(oh,2)))

# Specialized conversion utilities follow
gpu(x::Array) = x |> CuArrays.cu
gpu(x::AlignedBags) = CuAlignedBags(x)
# gpu(x::Flux.OneHotMatrix) = x |> Flux.gpu

# gpu(x::Flux.OneHotMatrix) = CuArrays.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC(x))
# gpu(x::NGramMatrix) = CuArrays.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC(x))

gpu(x::Flux.OneHotMatrix) = CuSparseMatrix(SparseMatrixCSC(x))
gpu(x::NGramMatrix) = CuSparseMatrix(SparseMatrixCSC(x))


# function gpu(x::NGramMatrix)
#     x_csc = SparseMatrixCSC(x)
#     x_sum = sum(x_csc)
#     if x_sum > 0.5
#         CuArrays.CUSPARSE.CuSparseMatrixCSC(x_csc ./ x_sum)
#     else
#         CuArrays.CUSPARSE.CuSparseMatrixCSC(x_csc)
#     end
# end
gpu(x::NamedTuple{KW}) where KW = NamedTuple{KW}(values(x) |> gpu)
gpu(x::Tuple) = gpu.(x)
gpu(x::Number) = x
gpu(x::Function) = x
gpu(x::Chain) = Chain((x.layers .|> gpu)...)

# The collision with Flux.gpu is intentional
#TODO: Perhaps the methods can be unified?
export gpu
