"""
    reflectinmodel(x::AbstractMillNode, fm=d -> Dense(d, 10), fa=BagCount ∘ SegmentedMeanMax;
        fsm=Dict(), fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false)

Build a [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) model capable of processing `x`.

All inner `Dense` layers are constructed using `fm`, a function accepting input dimension `d` and
returning a suitable model. All aggregation operators are constructed using `fa` in a similar manner.

More fine-grained control can be achieved with `fsm` and `fsa` keyword arguments, which should be
`Dict`s of `c => f` pairs, where `c` is a `String` traversal code from [`HierarchicalUtils.jl`](@ref) and
`f` is a function. These definitions override `fm` and `fa`.

If a [`ProductNode`](@ref) with only a single child (subtree) is encountered, its final `m` model
is instantiated as `identity` instead of using `fm` and `fsm`. This can be controlled with `single_key_identity`.

Similarly, if an [`ArrayNode`](@ref) contains data `X` where `size(X, 1)` is `1`, the corresponding
model is instantiated as `identity` unless `single_scalar_identity` is `false`.

By default, `reflectinmodel` makes first `Dense` layers in leafs imputing only if the datatype suggests
that missing data is present. This applies to

- `Array`,
- `SparseMatrixCSC` or `SparseVector`,
- `PooledArray`,
- [`MaybeHotVector`](@ref),
- [`MaybeHotMatrix`](@ref), and
- [`NGramMatrix`](@ref)

types with `eltype` of `{Union{Missing, T}} where T`. If `all_imputing` is true, all leaf `Dense`
layers in these types are replaced by their imputing variants.

# Examples
```jldoctest
julia> n1 = ProductNode(a=ArrayNode(NGramMatrix(["a", missing])))
ProductNode  2 obs, 0 bytes
  ╰── a: ArrayNode(2053×2 NGramMatrix with Union{Missing, Int64} elements)  2 obs, 97 bytes

julia> n2 = ProductNode((ArrayNode([0 1]), BagNode(ArrayNode([0 1; 2 3]), bags([1:1, 2:2]))))
ProductNode  2 obs, 0 bytes
  ├── ArrayNode(1×2 Array with Int64 elements)  2 obs, 72 bytes
  ╰── BagNode  2 obs, 80 bytes
        ╰── ArrayNode(2×2 Array with Int64 elements)  2 obs, 88 bytes

julia> n = ProductNode((n1, n2))
ProductNode  2 obs, 0 bytes
  ├── ProductNode  2 obs, 0 bytes
  │     ╰── a: ArrayNode(2053×2 NGramMatrix with Union{Missing, Int64} elements)  2 obs, 97 bytes
  ╰── ProductNode  2 obs, 0 bytes
        ├── ArrayNode(1×2 Array with Int64 elements)  2 obs, 72 bytes
        ╰── BagNode  2 obs, 80 bytes
              ╰── ArrayNode(2×2 Array with Int64 elements)  2 obs, 88 bytes

julia> reflectinmodel(n)
ProductModel ↦ Dense(20 => 10)  2 arrays, 210 params, 928 bytes
  ├── ProductModel ↦ identity
  │     ╰── a: ArrayModel([postimputing]Dense(2053 => 10))  3 arrays, 20_550 params, 80.398 KiB
  ╰── ProductModel ↦ Dense(11 => 10)  2 arrays, 120 params, 568 bytes
        ├── ArrayModel(identity)
        ╰── BagModel ↦ BagCount([SegmentedMean(10); SegmentedMax(10)]) ↦ Dense(21 => 10)  4 arrays, 240 params, 1.102 KiB
              ╰── ArrayModel(Dense(2 => 10))  2 arrays, 30 params, 208 bytes

julia> reflectinmodel(n, d -> Dense(d, 3), SegmentedMean, all_imputing=true)
ProductModel ↦ Dense(6 => 3)  2 arrays, 21 params, 172 bytes
  ├── ProductModel ↦ identity
  │     ╰── a: ArrayModel([postimputing]Dense(2053 => 3))  3 arrays, 6_165 params, 24.207 KiB
  ╰── ProductModel ↦ Dense(4 => 3)  2 arrays, 15 params, 148 bytes
        ├── ArrayModel([preimputing]Dense(1 => 1))  3 arrays, 3 params, 140 bytes
        ╰── BagModel ↦ SegmentedMean(3) ↦ Dense(3 => 3)  3 arrays, 15 params, 188 bytes
              ╰── ArrayModel([preimputing]Dense(2 => 3))  3 arrays, 11 params, 172 bytes

julia> printtree(n; trav=true)
ProductNode [""]  2 obs, 0 bytes
  ├── ProductNode ["E"]  2 obs, 0 bytes
  │     ╰── a: ArrayNode(2053×2 NGramMatrix with Union{Missing, Int64} elements) ["M"]  2 obs, 97 bytes
  ╰── ProductNode ["U"]  2 obs, 0 bytes
        ├── ArrayNode(1×2 Array with Int64 elements) ["Y"]  2 obs, 72 bytes
        ╰── BagNode ["c"]  2 obs, 80 bytes
              ╰── ArrayNode(2×2 Array with Int64 elements) ["e"]  2 obs, 88 bytes

julia> reflectinmodel(n, d -> Dense(d, 3), SegmentedMean;
                        fsm=Dict("e" => d -> Chain(Dense(d, 2), Dense(2, 2))),
                        fsa=Dict("c" => SegmentedLSE),
                        single_key_identity=false,
                        single_scalar_identity=false)
ProductModel ↦ Dense(6 => 3)  2 arrays, 21 params, 172 bytes
  ├── ProductModel ↦ Dense(3 => 3)  2 arrays, 12 params, 136 bytes
  │     ╰── a: ArrayModel([postimputing]Dense(2053 => 3))  3 arrays, 6_165 params, 24.207 KiB
  ╰── ProductModel ↦ Dense(6 => 3)  2 arrays, 21 params, 172 bytes
        ├── ArrayModel(Dense(1 => 3))  2 arrays, 6 params, 112 bytes
        ╰── BagModel ↦ SegmentedLSE(2) ↦ Dense(2 => 3)  4 arrays, 13 params, 220 bytes
              ╰── ArrayModel(Chain(Dense(2 => 2), Dense(2 => 2)))  4 arrays, 12 params, 224 bytes
```

See also: [`AbstractMillNode`](@ref), [`AbstractMillModel`](@ref), [`ProductNode`](@ref), [`BagNode`](@ref), [`ArrayNode`](@ref).
"""
function reflectinmodel(x, fm=d -> Dense(d, 10), fa=BagCount ∘ SegmentedMeanMax; fsm=Dict(),
        fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false)
    _reflectinmodel(x, fm, fa, fsm, fsa, "", single_key_identity, single_scalar_identity, all_imputing)[1]
end

function _reflectinmodel(x::AbstractBagNode, fm, fa, fsm, fsa, s, args...)
    c = stringify(s)
    im, d = _reflectinmodel(x.data, fm, fa, fsm, fsa, s * encode(1, 1), args...)
    agg = haskey(fsa, c) ? fsa[c](d) : fa(d)
    d = size(BagModel(im, agg)(x), 1)
    bm = haskey(fsm, c) ? fsm[c](d) : fm(d)
    m = BagModel(im, agg, bm)
    m, size(m(x), 1)
end

_remap(data::NamedTuple, ms) = (; zip(keys(data), ms)...)
_remap(::Tuple, ms) = tuple(ms...)
_remap(::AbstractVector, ms) = collect(ms)

function _reflectinmodel(x::AbstractProductNode, fm, fa, fsm, fsa, s, ski, args...)
    c = stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms, ds = zip([_reflectinmodel(x.data[k], fm, fa, fsm, fsa, s * encode(i, n), ski, args...)
                  for (i, k) in enumerate(ks)]...) |> collect
    ms = _remap(x.data, ms)
    m = if haskey(fsm, c)
        fsm[c](sum(ds))
    elseif ski && n == 1
        identity
    else
        fm(sum(ds))
    end
    m = ProductModel(ms, m)
    m, size(m(x), 1)
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
    m, size(m(x), 1)
end

function _reflectinmodel(ds::LazyNode{Name}, fm, fa, fsm, fsa, s, args...) where Name
    pm, d = _reflectinmodel(unpack2mill(ds), fm, fa, fsm, fsa, s * encode(1, 1), args...)
    LazyModel{Name}(pm), d
end

_make_imputing(x, t::ArrayModel, ai) = _make_imputing(x, t.m, ai) |> ArrayModel
_make_imputing(x, t::Chain, ai) = Chain(_make_imputing(x, t[1], ai), t[2:end]...)
_make_imputing(_, t, ai) = t

const ImputedArray{T} = Union{Array{T}, SparseMatrixCSC{T}, SparseVector{T}, PooledArray{T}}

_make_imputing(::ImputedArray, t::Dense, ai) = ai ? preimputing_dense(t) : t
_make_imputing(::ImputedArray{Maybe{T}}, t::Dense, ai) where T <: Number = preimputing_dense(t)
_make_imputing(x::ImputedArray, ::typeof(identity), ai) = ai ? preimputing_dense(_identity_dense(x)) : identity
_make_imputing(x::ImputedArray{Maybe{T}}, ::typeof(identity), ai) where T <: Number = preimputing_dense(_identity_dense(x))

_make_imputing(::NGramMatrix, t::Dense, ai) = ai ? postimputing_dense(t) : t
_make_imputing(::NGramMatrix{Maybe{T}}, t::Dense, ai) where T <: Sequence = postimputing_dense(t)
_make_imputing(x::NGramMatrix, ::typeof(identity), ai) = ai ? postimputing_dense(_identity_dense(x)) : identity
_make_imputing(x::NGramMatrix{Maybe{T}}, ::typeof(identity), ai) where T <: Sequence = postimputing_dense(_identity_dense(x))

_make_imputing(::MaybeHotArray, t::Dense, ai) = postimputing_dense(t)
_make_imputing(x::MaybeHotArray, ::typeof(identity), ai) = postimputing_dense(_identity_dense(x))

_identity_dense(x) = Dense(Matrix{Float32}(I, size(x, 1), size(x, 1)), zeros(Float32, size(x, 1)))
