using HierarchicalUtils: encode, stringify

abstract type AbstractMillModel end

const MillFunction = Union{Dense, Chain, Function}

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")

function Base.show(io::IO, @nospecialize(m::AbstractMillModel))
    print(io, nameof(typeof(m)))
    if !get(io, :compact, false)
        _show_submodels(io, m)
    end
end

_show_submodels(io, m::ArrayModel) = print(io, "(", m.m, ")")
_show_submodels(io, m::BagModel) = print(io, " … ↦ ", m.a, " ↦ ", m.bm)
_show_submodels(io, m::ProductModel) = print(io, " … ↦ ", m.m)
_show_submodels(io, m::LazyModel{Name}) where {Name} = print(io, "{", Name, "}")
_show_submodels(io, _) = print(io)

function reflectinmodel(x, fm=d->Flux.Dense(d, 10), fa=d->SegmentedMean(d); fsm=Dict(), fsa=Dict(),
               single_key_identity=true, single_scalar_identity=true)
    _reflectinmodel(x, fm, fa, fsm, fsa, "", single_key_identity, single_scalar_identity)[1]
end

function _reflectinmodel(x::AbstractBagNode, fm, fa, fsm, fsa, s, ski, ssi)
    c = stringify(s)
    im, d = _reflectinmodel(x.data, fm, fa, fsm, fsa, s * encode(1, 1), ski, ssi)
    agg = haskey(fsa, c) ? fsa[c](d) : fa(d)
    d = size(BagModel(im, agg)(x).data, 1)
    bm = haskey(fsm, c) ? fms[c](d) : fm(d)
    m = BagModel(im, agg, bm)
    m, size(m(x).data, 1)
end

_remap(data::NamedTuple, ms) = (; zip(keys(data), ms)...)
_remap(::Tuple, ms) = tuple(ms...)

function _reflectinmodel(x::AbstractProductNode, fm, fa, fsm, fsa, s, ski, ssi)
    c = stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms, ds = zip([_reflectinmodel(x.data[k], fm, fa, fsm, fsa, s * encode(i, n), ski, ssi)
                  for (i, k) in enumerate(ks)]...) |> collect
    ms = _remap(x.data, ms)
    m = if haskey(fsm, c)
        ArrayModel(fsm[c](sum(ds)))
    elseif ski && n == 1
        identity_model()
    else
        _reflectinmodel(ProductModel(ms)(x), fm, fa, fsm, fsa, s, ski, ssi)[1]
    end
    m = ProductModel(ms, m)
    m, size(m(x).data, 1)
end

function _reflectinmodel(x::ArrayNode, fm, fa, fsm, fsa, s, ski, ssi)
    c = stringify(s)
    r = size(x.data, 1)
    t = if haskey(fsm, c)
        fsm[c](r)
    elseif ssi && r == 1
        identity
    else
        fm(r)
    end |> ArrayModel
    m = _make_imputing(x.data, t)
    m, size(m(x).data, 1)
end

_make_imputing(x, t) = t
_make_imputing(x, t::ArrayModel) = _make_imputing(x, t.m) |> ArrayModel
_make_imputing(x, t::Chain) = Chain(_make_imputing(x, t[1]), t[2:end]...)
_make_imputing(x::AbstractArray{Maybe{T}}, t::Dense) where T <: Number = preimputing_dense(t)
_make_imputing(x::MaybeHotVector{Missing}, t::Dense) = postimputing_dense(t)
_make_imputing(x::MaybeHotMatrix{Maybe{T}}, t::Dense) where T <: Integer = postimputing_dense(t)
_make_imputing(x::NGramMatrix{Maybe{T}}, t::Dense) where T <: Sequence = postimputing_dense(t)

identity_dense(x) = Dense(Matrix{Float32}(I, size(x, 1), size(x, 1)), zeros(Float32, size(x, 1)))

_make_imputing(x, t::typeof(identity)) = t
function _make_imputing(x::AbstractArray{Maybe{T}}, ::typeof(identity)) where T <: Number
    preimputing_dense(identity_dense(x))
end
function _make_imputing(x::MaybeHotVector{Missing}, ::typeof(identity))
    postimputing_dense(identity_dense(x))
end
function _make_imputing(x::MaybeHotMatrix{Maybe{T}}, ::typeof(identity)) where T <: Integer
    postimputing_dense(identity_dense(x))
end
function _make_imputing(x::NGramMatrix{Maybe{T}}, ::typeof(identity)) where T <: Sequence
    postimputing_dense(identity_dense(x))
end

function _reflectinmodel(ds::LazyNode{Name}, fm, fa, fsm, fsa, s, ski, ssi) where Name
    pm, d = Mill._reflectinmodel(unpack2mill(ds), fm, fa, fsm, fsa, s * Mill.encode(1, 1), ski, ssi)
    LazyModel{Name}(pm), d
end
