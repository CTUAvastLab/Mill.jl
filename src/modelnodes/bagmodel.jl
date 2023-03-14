"""
    BagModel{T <: AbstractMillModel, A <: Union{AbstractAggregation, BagCount}, U}
        <: AbstractMillModel

A model node for processing [`AbstractBagNode`](@ref)s. It first applies its \"instance (sub)model\" `im`
on every instance, then performs elementwise segmented aggregation `a` and finally applies the final
model `bm` on the aggregated representation of every bag in the data node.

# Examples
```jldoctest; filter=$(DOCTEST_FILTER)
julia> Random.seed!(0);

julia> n = BagNode(ArrayNode(randn(Float32, 3, 2)), bags([0:-1, 1:2]))
BagNode  # 2 obs, 96 bytes
  ╰── ArrayNode(3×2 Array with Float32 elements)  # 2 obs, 72 bytes

julia> m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMeanMax(2), Dense(4, 2))
BagModel ↦ [SegmentedMean(2); SegmentedMax(2)] ↦ Dense(4 => 2)  # 4 arrays, 14 params, 216 bytes
  ╰── ArrayModel(Dense(3 => 2))  # 2 arrays, 8 params, 112 bytes

julia> m(n)
2×2 Matrix{Float32}:
 0.0  1.05...
 0.0  0.49...

julia> m(n) == m.bm(m.a(m.im(n.data), n.bags))
true
```

See also: [`AbstractMillModel`](@ref), [`AbstractAggregation`](@ref), [`AbstractBagNode`](@ref),
    [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
struct BagModel{T <: AbstractMillModel, A <: Union{AbstractAggregation, BagCount},
                U} <: AbstractMillModel
    im::T
    a::A
    bm::U
end

Flux.@functor BagModel

"""
    BagModel(im, a, bm=identity)

Construct a [`BagModel`](@ref) from the arguments. `im` should be [`AbstractMillModel`](@ref),
`a` [`AbstractAggregation`](@ref) or [`BagCount`](@ref), and `bm` [`ArrayModel`](@ref).

It is also possible to pass any function as `im` instead of a model node. In that case,
it is wrapped into an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> m = BagModel(ArrayModel(Dense(3, 2)), SegmentedMeanMax(2), Dense(4, 2))
BagModel ↦ [SegmentedMean(2); SegmentedMax(2)] ↦ Dense(4 => 2)  # 4 arrays, 14 params, 216 bytes
  ╰── ArrayModel(Dense(3 => 2))  # 2 arrays, 8 params, 112 bytes

julia> m = BagModel(Dense(4, 3), BagCount(SegmentedMean(3)))
BagModel ↦ BagCount(SegmentedMean(3)) ↦ identity  # 1 arrays, 3 params (all zero), 52 bytes
  ╰── ArrayModel(Dense(4 => 3))  # 2 arrays, 15 params, 140 bytes
```

See also: [`AbstractMillModel`](@ref), [`AbstractAggregation`](@ref), [`BagCount`](@ref),
    [`AbstractBagNode`](@ref), [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
BagModel(im, a::Union{AbstractAggregation, BagCount}, bm=identity) = BagModel(_arraymodel(im), a, bm)

(m::BagModel)(x::BagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::BagNode{Missing}) = m.bm(m.a(x.data, x.bags))
(m::BagModel)(x::WeightedBagNode{<:AbstractMillNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))
(m::BagModel)(x::WeightedBagNode{Missing}) = m.bm(m.a(x.data, x.bags, x.weights))
