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

# Specialized conversion utilities follow
gpu(x::Array) = x |> CuArrays.cu
gpu(x::AlignedBags) = CuAlignedBags(x)
gpu(x::Flux.OneHotMatrix) = x |> Flux.gpu
gpu(x::NGramMatrix) = CuArrays.CUSPARSE.CuSparseMatrixCSC(SparseMatrixCSC(x))
gpu(x::NamedTuple{KW}) where KW = NamedTuple{KW}(values(x) |> gpu)
gpu(x::Tuple) = gpu.(x)
gpu(x::Number) = x
gpu(x::Function) = x
gpu(x::Chain) = Chain((x.layers .|> gpu)...)

# The collision with Flux.gpu is intentional
#TODO: Perhaps the methods can be unified?
export gpu
