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
    w::Vector{Float32}
end

Flux.@functor ProductModel

ProductModel(m::AbstractMillModel) = ProductModel((m,))
ProductModel(ms::TT) where {TT<:TupleOfModels} = ProductModel(ms, identity_model())
ProductModel(ms, f::MillFunction) = ProductModel(ms, ArrayModel(f))
ProductModel(ms::TupleOfModels, m::ArrayModel) = ProductModel(ms, m, ones(Float32, length(ms)))
  
Base.getindex(m::ProductModel, i::Symbol) = m.ms[i]
Base.keys(m::ProductModel{P,T}) where {P<:NamedTuple,T} = keys(m.ms)
Base.keys(m::ProductModel{P,T}) where {P<:Tuple,T} = 1:length(m.ms)

# function (m::ProductModel)(x::ProductNode)
#     xs = ThreadsX.map(k -> m.ms[k](x.data[k]), keys(m))
#     m.m(vcat(xs...))
# end

function (m::ProductModel)(x::ProductNode)
    xs = ThreadsX.map((i, k) -> ArrayNode(m.w[i] .* (m.ms[k](x.data[k])).data), enumerate(keys(m)))
    m.m(vcat(xs...))
end
