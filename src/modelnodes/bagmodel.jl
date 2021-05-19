"""
    BagModel{T <: AbstractMillModel, A <: Union{AbstractAggregation, BagCount}, U <: ArrayModel}
        <: AbstractMillModel

A model node for processing [`AbstractBagNode`](@ref)s. It first applies its \"instance (sub)model\" `im`
on every instance, then performs elementwise segmented aggregation `a` and finally applies the final
model `bm` on the aggregated representation of every bag in the data node.

# Examples
```jldoctest; filter=$(DOCTEST_FILTER)
julia> Random.seed!(0);

julia> n = BagNode(ArrayNode(randn(3, 2)), bags([0:-1, 1:2]))
BagNode with 2 obs
  └── ArrayNode(3×2 Array with Float64 elements) with 2 obs

julia> m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMeanMax(2), ArrayModel(Dense(4, 2)))
BagModel … ↦ [SegmentedMean(2); SegmentedMax(2)] ↦ ArrayModel(Dense(4, 2))
  └── ArrayModel(Dense(3, 2))

julia> m(n)
2×2 ArrayNode{Matrix{Float64}, Nothing}:
 0.0  -2.049...
 0.0  -0.906...

julia> m.bm(m.a(m.im(n.data), n.bags))
2×2 ArrayNode{Matrix{Float64}, Nothing}:
 0.0  -2.049...
 0.0  -0.906...
```

See also: [`AbstractMillModel`](@ref), [`AbstractAggregation`](@ref), [`AbstractBagNode`](@ref),
    [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
struct BagModel{T <: AbstractMillModel, A <: Union{AbstractAggregation, BagCount},
                U <: ArrayModel} <: AbstractMillModel
    im::T
    a::A
    bm::U
end

Flux.@functor BagModel

"""
    BagModel(im, a, bm=identity_model())

Construct a [`BagModel`](@ref) from the arguments. `im` should be [`AbstractMillModel`](@ref),
`a` [`AbstractAggregation`](@ref) or [`BagCount`](@ref), and `bm` [`ArrayModel`](@ref).

It is also possible to pass any function (`Flux.Dense`, `Flux.Chain`, `identity`...) as `im` or `bm`.
In that case, they are wrapped into an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMeanMax(2), ArrayModel(Dense(4, 2)))
BagModel … ↦ [SegmentedMean(2); SegmentedMax(2)] ↦ ArrayModel(Dense(4, 2))
  └── ArrayModel(Dense(3, 2))

julia> m = BagModel(Dense(4, 3), BagCount(SegmentedMean(3)))
BagModel … ↦ BagCount(SegmentedMean(3)) ↦ ArrayModel(identity)
  └── ArrayModel(Dense(4, 3))
```

See also: [`AbstractMillModel`](@ref), [`AbstractAggregation`](@ref), [`BagCount`](@ref),
    [`AbstractBagNode`](@ref), [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
function BagModel(im::Union{MillFunction, AbstractMillModel}, a::Union{AbstractAggregation, BagCount},
        bm::Union{MillFunction, ArrayModel}=identity_model())
    BagModel(_make_array_model(im), a, _make_array_model(bm))
end

(m::BagModel)(x::BagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::BagNode{Missing}) = m.bm(ArrayNode(m.a(x.data, x.bags)))
(m::BagModel)(x::WeightedBagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
(m::BagModel)(x::WeightedBagNode{Missing}) = m.bm(ArrayNode(m.a(x.data, x.bags, x.weights)))

function HiddenLayerModel(m::BagModel, x::BagNode, k::Int)
    im, o = HiddenLayerModel(m.im, x.data, k)
    a = SegmentedMax(k)
    b = m.a(o, x.bags)
    bm, o = HiddenLayerModel(m.bm, b, k)
    BagModel(im, a, bm), o
end

function mapactivations(hm::BagModel, x::BagNode{<:AbstractMillNode}, m::BagModel)
    hmi, mi = mapactivations(hm.im, x.data, m.im)
    ai = m.a(mi, x.bags)
    hai = hm.a(hmi, x.bags)
    hbo, bo = mapactivations(hm.bm, ai, m.bm)
    (ArrayNode(hbo.data + hai.data), bo)
end

function mapactivations(hm::BagModel, x::BagNode{<:Missing}, m::BagModel)
    ai = m.a(missing, x.bags)
    hai = hm.a(missing, x.bags)
    hbo, bo = mapactivations(hm.bm, ArrayNode(ai), m.bm)
    (ArrayNode(hbo.data + hai), bo)
end

function fold(f, m::BagModel, x)
    o₁ = fold(f, m.im, x.data)
    o₂ = f(m.a, o₁, x.bags)
    o₃ = fold(f, m.bm, o₂)
    o₃
end

# Base.hash(m::BagModel{T,A,U}, h::UInt) where {T,A,U} = hash((T, A, U, m.im, m.a, m.bm), h)
# (m1::BagModel{T,A,U} == m2::BagModel{T,A,U}) where {T,A,U} =
    # m1.im == m2.im && m1.a == m2.a && m1.bm == m2.bm
