const TupleOfModels = Union{NTuple{N, MillModel} where {N}, NamedTuple}

"""
    struct ProductModel{N, T <: MillFunction} <: MillModel
        ms::NTuple{N, MillModel}
        m::ArrayModel{T}
    end

    uses each model in `ms` on each data in `TreeNode`, concatenate the output and pass it to the chainmodel `m`
"""
struct ProductModel{TT<:TupleOfModels, T} <: MillModel
    ms::TT
    m::ArrayModel{T}
end

Flux.@functor ProductModel

ProductModel(ms::TT) where {TT<:TupleOfModels} = ProductModel(ms, ArrayModel(identity))
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))

function (m::ProductModel{MS,M})(x::TreeNode{P,T}) where {P<:Tuple,T,MS<:Tuple, M} 
    xx = vcat([m.ms[i](x.data[i]) for i in 1:length(m.ms)]...)
    m.m(xx)
end

function (m::ProductModel{MS,M})(x::TreeNode{P,T}) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    xx = vcat([m.ms[k](x.data[k]) for k in keys(m.ms)]...)
    m.m(xx)
end

function modelprint(io::IO, m::ProductModel; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductModel$(tr_repr(s, tr)) (", color=c)

    n = length(m.ms)
    ks = key_labels(m.ms)
    for i in 1:(n-1)
        println(io)
        paddedprint(io, "  ├── $(ks[i])", color=c, pad=pad)
        modelprint(io, m.ms[i], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(ks[i]))))], s=s * encode(i, n), tr=tr)
    end
    println(io)
    paddedprint(io, "  └── $(ks[end])", color=c, pad=pad)
    modelprint(io, m.ms[end], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(ks[end]))))], s=s * encode(n, n), tr=tr)

    println(io)
    paddedprint(io, " ) ↦  ", color=c, pad=pad)
    modelprint(io, m.m, pad=[pad; (c, "")])
end



function HiddenLayerModel(m::ProductModel, x::TreeNode, k::Int)
    ks = keys(m.ms)
    hxms = [HiddenLayerModel(m.ms[i], x.data[i], k) for i in keys(m.ms)]
    hms = (;[ks[i] => hxms[i][1] for i in 1:length(ks)]...)
    xms = vcat([hxms[i][2] for i in 1:length(ks)]...)

    hm, o = HiddenLayerModel(m.m, xms, k)
    ProductModel(hms, hm), o
end


function mapactivations(hm::ProductModel, x::TreeNode, m::ProductModel)
    ks = keys(m.ms)
    _xxs = [mapactivations(hm.ms[i], x.data[i], m.ms[i]) for i in keys(m.ms)]
    hxs = foldl( +, [_xxs[i][1].data for i in 1:length(ks)])
    xxs = vcat([_xxs[i][2] for i in 1:length(ks)]...)

    ho, o = mapactivations(hm.m, xxs, m.m)
    (ArrayNode(ho.data + hxs), o)
end