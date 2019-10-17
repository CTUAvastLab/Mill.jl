const TupleOfModels = Union{NTuple{N, MillModel} where {N}, NamedTuple}

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

function (m::ProductModel)(x::TreeNode) 
    xx = vcat([m.ms[i](x.data[i]) for i in 1:length(m.ms)]...)
    m.m(xx)
end

function modelprint(io::IO, m::ProductModel; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductModel$(tr_repr(s, tr)) (", color=c)

    n = length(m.ms)
    ks = key_labels(m.ms)
    for i in 1:(n-1)
        println()
        paddedprint(io, "  ├── $(ks[i])", color=c, pad=pad)
        modelprint(io, m.ms[i], pad=[pad; (c, "  │   ")], s=s * encode(i, n), tr=tr)
    end
    println()
    paddedprint(io, "  └── $(ks[end])", color=c, pad=pad)
    modelprint(io, m.ms[end], pad=[pad; (c, "      ")], s=s * encode(n, n), tr=tr)

    println()
    paddedprint(io, " ) ↦  ", color=c, pad=pad)
    modelprint(io, m.m, pad=[pad; (c, "")])
end
