"""
    BagModel{T <: AbstractMillModel, A <: Aggregation, U <: ArrayModel} <: AbstractMillModel

A model node for processing [`AbstractBagNode`](@ref)s. It first applies its \"instance (sub)model\" `im`
on every instance, then performs elementwise segmented aggregation `a` and finally applies the final
model `bm` on the aggregated representation of every bag in the data node.

# Examples
```jldoctest
julia> Random.seed!(0);

julia> n = BagNode(ArrayNode(randn(2, 2)), bags([0:-1, 1:2]))
BagNode with 2 obs
  └── ArrayNode(2×2 Array with Float64 elements) with 2 obs

julia> m = BagModel(ArrayModel(Dense(2, 2)), meanmax_aggregation(2), ArrayModel(Dense(5, 2)))
BagModel … ↦ ⟨SegmentedMean(2), SegmentedMax(2)⟩ ↦ ArrayModel(Dense(5, 2))
  └── ArrayModel(Dense(2, 2))

julia> m(n)
2×2 ArrayNode{Array{Float32,2},Nothing}:
 0.0  -1.1958722
 0.0   0.62269455

julia> m.bm(m.a(m.im(n.data), n.bags))
2×2 ArrayNode{Array{Float32,2},Nothing}:
 0.0  -1.1958722
 0.0   0.62269455
```

See also: [`AbstractMillModel`](@ref), [`Aggregation`](@ref), [`AbstractBagNode`](@ref),
    [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
struct BagModel{T <: AbstractMillModel, A <: Aggregation, U <: ArrayModel} <: AbstractMillModel
    im::T
    a::A
    bm::U
end

Flux.@functor BagModel

"""
    BagModel(im, a, bm=identity_model())

Construct a [`BagModel`](@ref) from the arguments. `im` should [`AbstractMillModel`](@ref),
`a` [`Aggregation`](@ref), and `bm` [`ArrayModel`](@ref).

It is also possible to pass any function (`Flux.Dense`, `Flux.Chain`, `identity`...) as `im` or `bm`.
In that case, they are wrapped into an [`ArrayNode`](@ref).

# Examples
```jldoctest
julia> m = BagModel(ArrayModel(Dense(2, 3)), max_aggregation(2), ArrayModel(Dense(3, 2)))
BagModel … ↦ ⟨SegmentedMax(2)⟩ ↦ ArrayModel(Dense(3, 2))
  └── ArrayModel(Dense(2, 3))

julia> m = BagModel(Dense(2, 3), mean_aggregation(2))
BagModel … ↦ ⟨SegmentedMean(2)⟩ ↦ ArrayModel(identity)
  └── ArrayModel(Dense(2, 3))
```

See also: [`AbstractMillModel`](@ref), [`Aggregation`](@ref), [`AbstractBagNode`](@ref),
    [`BagNode`](@ref), [`WeightedBagNode`](@ref).
"""
function BagModel(im::Union{MillFunction, AbstractMillModel}, a::Aggregation,
        bm::Union{MillFunction, ArrayModel}=identity_model())
    BagModel(_make_array_model(im), a, _make_array_model(bm))
end

(m::BagModel)(x::WeightedBagNode{<: AbstractNode}) = m.bm(m.a(m.im(x.data), x.bags, x.weights))

(m::BagModel)(x::BagNode) = m.bm(m.a(m.im(x.data), x.bags))
(m::BagModel)(x::BagNode{Missing}) = m.bm(ArrayNode(m.a(x.data, x.bags)))

function HiddenLayerModel(m::BagModel, x::BagNode, k::Int)
    im, o = HiddenLayerModel(m.im, x.data, k)
    a = max_aggregation(k)
    b = m.a(o, x.bags)
    bm, o = HiddenLayerModel(m.bm, b, k+1)
    BagModel(im, a, bm), o
end

function mapactivations(hm::BagModel, x::BagNode{M,B,C}, m::BagModel) where {M<:AbstractNode,B,C}
    hmi, mi = mapactivations(hm.im, x.data, m.im)
    ai = m.a(mi, x.bags)
    hai = hm.a(hmi, x.bags)
    hbo, bo = mapactivations(hm.bm, ai, m.bm)
    (ArrayNode(hbo.data + hai.data), bo)
end

function mapactivations(hm::BagModel, x::BagNode{M,B,C}, m::BagModel) where {M<:Missing,B,C}
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
