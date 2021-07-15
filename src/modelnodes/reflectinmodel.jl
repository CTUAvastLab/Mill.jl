"""
    reflectinmodel(x::AbstractMillNode, fm=d -> Dense(d, 10), fa=BagCount ∘ SegmentedMeanMax;
        fsm=Dict(), fsa=Dict(), single_key_identity=true, single_scalar_identity=true)

Build a `Mill.jl` model capable of processing `x`.

All inner `Dense` layers are constructed using `fm`, a function accepting input dimension `d` and
returning suitable model. All aggregation operators are constructed using `fa` in a similar manner.

More fine-grained control can be achieved with `fsm` and `fsa` keyword arguments, which should be
`Dict`s of `c => f` pairs, where `c` is a `String` traversal code from [HierarchicalUtils.jl](@ref) and
`f` is a function. These definitions override `fm` and `fa`.

If a [`ProductNode`](@ref) with only a single child (subtree) is encountered, its final `m` model
is instantiated as `identity` instead of using `fm` and `fsm`. This can be controlled with `single_key_identity`.

Similarly, if an [`ArrayNode`](@ref) contains data `X` where `size(X, 1)` is `1`, the corresponding
model is instantiated as `identity` unless `single_scalar_identity` is `false`.

# Examples
```jldoctest
julia> n1 = ProductNode((; a=ArrayNode(NGramMatrix(["a", "b"]))))
ProductNode with 2 obs
  └── a: ArrayNode(2053×2 NGramMatrix with Int64 elements)

julia> n2 = ProductNode((ArrayNode([0 1]), BagNode(ArrayNode([0 1; 2 3]), bags([1:1, 2:2]))))
ProductNode with 2 obs
  ├── ArrayNode(1×2 Array with Int64 elements)
  └── BagNode with 2 obs
        └── ArrayNode(2×2 Array with Int64 elements)

julia> n = ProductNode((n1, n2))
ProductNode with 2 obs
  ├── ProductNode with 2 obs
  │     └── a: ArrayNode(2053×2 NGramMatrix with Int64 elements)
  └── ProductNode with 2 obs
        ├── ArrayNode(1×2 Array with Int64 elements)
        └── BagNode with 2 obs
              ⋮

julia> printtree(n; trav=true)
ProductNode with 2 obs [""]
  ├── ProductNode with 2 obs ["E"]
  │     └── a: ArrayNode(2053×2 NGramMatrix with Int64 elements) ["M"]
  └── ProductNode with 2 obs ["U"]
        ├── ArrayNode(1×2 Array with Int64 elements) ["Y"]
        └── BagNode with 2 obs ["c"]
              └── ArrayNode(2×2 Array with Int64 elements) ["e"]

julia> reflectinmodel(n) |> printtree
ProductModel … ↦ ArrayModel(Dense(20, 10))
  ├── ProductModel … ↦ ArrayModel(identity)
  │     └── a: ArrayModel(Dense(2053, 10))
  └── ProductModel … ↦ ArrayModel(Dense(11, 10))
        ├── ArrayModel(identity)
        └── BagModel … ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ ArrayModel(Dense(21, 10))
              └── ArrayModel(Dense(2, 10))

julia> reflectinmodel(n, d -> Dense(d, 3), SegmentedMean) |> printtree
ProductModel … ↦ ArrayModel(Dense(6, 3))
  ├── ProductModel … ↦ ArrayModel(identity)
  │     └── a: ArrayModel(Dense(2053, 3))
  └── ProductModel … ↦ ArrayModel(Dense(4, 3))
        ├── ArrayModel(identity)
        └── BagModel … ↦ SegmentedMean(3) ↦ ArrayModel(Dense(3, 3))
              └── ArrayModel(Dense(2, 3))

julia> reflectinmodel(n, d -> Dense(d, 3), SegmentedMean;
                        fsm=Dict("e" => d -> Chain(Dense(d, 2), Dense(2, 2))),
                        fsa=Dict("c" => SegmentedLSE),
                        single_key_identity=false,
                        single_scalar_identity=false) |> printtree
ProductModel … ↦ ArrayModel(Dense(6, 3))
  ├── ProductModel … ↦ ArrayModel(Dense(3, 3))
  │     └── a: ArrayModel(Dense(2053, 3))
  └── ProductModel … ↦ ArrayModel(Dense(6, 3))
        ├── ArrayModel(Dense(1, 3))
        └── BagModel … ↦ SegmentedLSE(2) ↦ ArrayModel(Dense(2, 3))
              └── ArrayModel(Chain(Dense(2, 2), Dense(2, 2)))
```

See also: [`AbstractMillNode`](@ref), [`AbstractMillModel`](@ref), [`ProductNode`](@ref), [`ArrayNode`](@ref).
"""
function reflectinmodel(x, fm=d -> Dense(d, 10), fa=BagCount ∘ SegmentedMeanMax; fsm=Dict(),
        fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false)
    _reflectinmodel(x, fm, fa, fsm, fsa, "", single_key_identity, single_scalar_identity, all_imputing)[1]
end

function _reflectinmodel(x::AbstractBagNode, fm, fa, fsm, fsa, s, args...)
    c = stringify(s)
    im, d = _reflectinmodel(x.data, fm, fa, fsm, fsa, s * encode(1, 1), args...)
    agg = haskey(fsa, c) ? fsa[c](d) : fa(d)
    d = size(BagModel(im, agg)(x).data, 1)
    bm = haskey(fsm, c) ? fsm[c](d) : fm(d)
    m = BagModel(im, agg, bm)
    m, size(m(x).data, 1)
end

_remap(data::NamedTuple, ms) = (; zip(keys(data), ms)...)
_remap(::Tuple, ms) = tuple(ms...)

function _reflectinmodel(x::AbstractProductNode, fm, fa, fsm, fsa, s, ski, args...)
    c = stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms, d = zip([_reflectinmodel(x.data[k], fm, fa, fsm, fsa, s * encode(i, n), ski, args...)
                  for (i, k) in enumerate(ks)]...) |> collect
    ms = _remap(x.data, ms)
    m = if haskey(fsm, c)
        ArrayModel(fsm[c](sum(d)))
    elseif ski && n == 1
        identity_model()
    else
        _reflectinmodel(ProductModel(ms)(x), fm, fa, fsm, fsa, s, ski, args...)[1]
    end
    m = ProductModel(ms, m)
    m, size(m(x).data, 1)
end

function _reflectinmodel(x::ArrayNode, fm, fa, fsm, fsa, s, ski, ssi, ai)
    c = stringify(s)
    r = size(x.data, 1)
    m = if haskey(fsm, c)
        fsm[c](r)
    elseif ssi && r == 1
        identity
    else
        fm(r)
    end |> ArrayModel
    m = _make_imputing(x.data, m, ai)
    m, size(m(x).data, 1)
end

function _reflectinmodel(ds::LazyNode{Name}, fm, fa, fsm, fsa, s, args...) where Name
    pm, d = _reflectinmodel(unpack2mill(ds), fm, fa, fsm, fsa, s * encode(1, 1), args...)
    LazyModel{Name}(pm), d
end

_make_imputing(x, t::ArrayModel, ai) = _make_imputing(x, t.m, ai) |> ArrayModel
_make_imputing(x, t::Chain, ai) = Chain(_make_imputing(x, t[1], ai), t[2:end]...)

_make_imputing(x::AbstractArray, t::Dense, ai) = ai ? preimputing_dense(t) : t
_make_imputing(x::AbstractArray{Maybe{T}}, t::Dense, ai) where T <: Number = preimputing_dense(t)
_make_imputing(x::AbstractArray, ::typeof(identity), ai) = ai ? preimputing_dense(_identity_dense(x)) : identity
_make_imputing(x::AbstractArray{Maybe{T}}, ::typeof(identity), ai) where T <: Number = preimputing_dense(_identity_dense(x))

_make_imputing(x::NGramMatrix, t::Dense, ai) = ai ? postimputing_dense(t) : t
_make_imputing(x::NGramMatrix{Maybe{T}}, t::Dense, ai) where T <: Sequence = postimputing_dense(t)
_make_imputing(x::NGramMatrix, ::typeof(identity), ai) = ai ? postimputing_dense(_identity_dense(x)) : identity
_make_imputing(x::NGramMatrix{Maybe{T}}, ::typeof(identity), ai) where T <: Sequence = postimputing_dense(_identity_dense(x))

_make_imputing(x::MaybeHotArray, t::Dense, ai) = postimputing_dense(t)
_make_imputing(x::MaybeHotArray, ::typeof(identity), ai) = postimputing_dense(_identity_dense(x))

_identity_dense(x) = Dense(Matrix{Float32}(I, size(x, 1), size(x, 1)), zeros(Float32, size(x, 1)))
