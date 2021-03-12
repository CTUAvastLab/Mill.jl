const TupleOfModels = Union{NTuple{N, AbstractMillModel} where {N}, NamedTuple}

"""
    struct ProductModel{N, T <: MillFunction} <: AbstractMillModel
        ms::NTuple{N, AbstractMillModel}
        m::ArrayModel{T}
    end

    uses each model in `ms` on each data in `ProductNode`, concatenate the output and pass it to the chainmodel `m`
"""
struct ProductModel{TT<:TupleOfModels, T} <: AbstractMillModel
    ms::TT
    m::ArrayModel{T}
end

Flux.@functor ProductModel

ProductModel(m::AbstractMillModel) = ProductModel((m,))
ProductModel(ms::TT) where {TT<:TupleOfModels} = ProductModel(ms, identity_model())
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))

Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel{P,T}) where {P<:NamedTuple,T} = keys(m.ms)
Base.keys(m::ProductModel{P,T}) where {P<:Tuple,T} = 1:length(m.ms)

function (m::ProductModel{MS,M})(x::ProductNode{P,T}) where {P<:Tuple,T,MS<:Tuple, M} 
    xx = ArrayNode(vcat([m.ms[i](x.data[i]) |> data for i in 1:length(m.ms)]...))
    m.m(xx)
end

function (m::ProductModel{MS,M})(x::ProductNode{P,T}) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    ks = Zygote.@ignore collect(keys(m.ms))
    if length(ks) == 1 
        child_m, child_ds = only(m.ms), only(x.data)
        return(m.m(child_m(child_ds)))
    end
    # xs = ThreadPools.qmap(k -> m.ms[k](x.data[k]).data, ks)
    # xs = ThreadTools.tmap(k -> m.ms[k](x.data[k]).data, ks)
    xs = ThreadsX.map(k -> m.ms[k](x.data[k]).data, ks)
    # xs = map(k -> m.ms[k](x.data[k]).data, ks)
    # xx = ArrayNode(vcat(xs...))
    xx = ArrayNode(VCatView(tuple(xs...)))
    m.m(xx)
end
