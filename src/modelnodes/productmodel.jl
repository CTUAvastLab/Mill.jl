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
    xx = ThreadsX.map(i -> m.ms[i](x.data[i]), keys(m))
    # xx = map(i -> m.ms[i](x.data[i]), keys(m))
    m.m(vcat(xx...))
end



function ∇tmap(cx, f, args...)
    ys_and_backs = ThreadsX.map((args...) -> Zygote._pullback(cx, f, args...), args...)
    if isempty(ys_and_backs)
      ys_and_backs, _ -> nothing
    else
      ys, backs = Zygote.unzip(ys_and_backs)
      ys, function (Δ)
        # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
        Δf_and_args_zipped = ThreadsX.map((f, δ) -> f(δ), Zygote._tryreverse(ThreadsX.map, backs, Δ)...)
        Δf_and_args = Zygote.unzip(Zygote._tryreverse(ThreadsX.map, Δf_and_args_zipped))
        Δf = reduce(Zygote.accum, Δf_and_args[1])
        (Δf, Δf_and_args[2:end]...)
      end
    end
end

Zygote.@adjoint function ThreadsX.map(f, args::Union{AbstractArray,Tuple}...)
    ∇tmap(__context__, f, args...)
end
