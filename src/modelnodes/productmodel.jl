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

function (m::ProductModel)(x::ProductNode)
    xs = ThreadsX.map(i -> m.ms[i](x.data[i]), keys(m))
    # xs = ThreadTools.tmap(i -> m.ms[i](x.data[i]), keys(m))
    m.m(vcat(xs...))
end