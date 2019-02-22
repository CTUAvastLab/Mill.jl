const TupleOfModels = NTuple{N, MillModel} where {N}

"""
    struct ProductModel{N, T <: MillFunction} <: MillModel
        ms::NTuple{N, MillModel}
        m::ArrayModel{T}
    end

    uses each model in `ms` on each data in `TreeNode`, concatenate the output and pass it to the chainmodel `m`
"""
struct ProductModel{TT<:TupleOfModels, T <: MillFunction} <: MillModel
    ms::TT
    m::ArrayModel{T}
end

Flux.@treelike ProductModel

ProductModel(ms::TT) where {TT<:TupleOfModels} = ProductModel(ms, ArrayModel(identity))
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))

(m::ProductModel)(x::TreeNode) = m.m(ArrayNode(vcat(map(f -> f[1](f[2]).data, zip(m.ms, x.data))...)))

function modelprint(io::IO, m::ProductModel; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductModel$(repr(s, tr)) (\n", color=c)

    n = length(m.ms)
    for i in 1:(n-1)
        paddedprint(io, "  ├── ", color=c, pad=pad)
        modelprint(io, m.ms[i], pad=[pad; (c, "  │   ")], s=encode(s, i, n), tr=tr)
    end
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.ms[end], pad=[pad; (c, "      ")], s=encode(s, n, n), tr=tr)

    paddedprint(io, ") ↦  ", color=c, pad=pad)
    modelprint(io, m.m, pad=[pad; (c, "")])
end
