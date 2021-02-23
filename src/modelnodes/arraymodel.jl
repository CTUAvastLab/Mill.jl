"""
    ArrayModel{T <: MillFunction} <: AbstractMillModel

A model node for processing [`ArrayNode`](@ref)s. It applies a (sub)model `m` stored in it to data of 
the [`ArrayNode`](@ref).

# Examples
```jldoctest; filter=r"-?[0-9]+\\.[0-9]+"
julia> Random.seed!(0);

julia> n = ArrayNode(randn(Float32, 2, 2))
2×2 ArrayNode{Array{Float32,2},Nothing}:
 0.6791074  -0.3530074
 0.8284135  -0.13485387

julia> m = ArrayModel(Dense(2, 2))
ArrayModel(Dense(2, 2))

julia> m(n)
2×2 ArrayNode{Array{Float32,2},Nothing}:
 0.660979    -0.18795347
 0.10059327   0.27500954
```

See also: [`AbstractMillModel`](@ref), [`IdentityModel`](@ref), [`identity_model`](@ref), [`ArrayNode`](@ref).
"""
struct ArrayModel{T <: MillFunction} <: AbstractMillModel
    m::T
end

Flux.@functor ArrayModel

(m::ArrayModel)(x::ArrayNode) = ArrayNode(m.m(x.data))

"""
    identity_model()

Returns an [`ArrayModel`](@ref) realising the `identity` transformation.

# Examples
```jldoctest
julia> identity_model()
ArrayModel(identity)
```

See also: [`ArrayModel`](@ref), [`IdentityModel`](@ref).
"""
identity_model() = ArrayModel(identity)

"""
    IdentityModel

Alias for `ArrayModel{typeof(identity)}`.

See also: [`ArrayModel`](@ref), [`identity_model`](@ref).
"""
const IdentityModel = ArrayModel{typeof(identity)}

function HiddenLayerModel(m::ArrayModel, x::ArrayNode, k::Int)
    os = Flux.activations(m.m, x.data)
    layers = Chain(map(x -> Dense(size(x,1), k), os)...)
    ArrayModel(layers), ArrayNode(os[end])
end

function mapactivations(hm::ArrayModel, x::ArrayNode, m::ArrayModel)
    os = Flux.activations(m.m, x.data)
    hx = mapfoldl((mx) -> mx[1](mx[2]),+,zip(hm.m, os))
    (ArrayNode(hx), ArrayNode(os[end]))
end

fold(f, m::ArrayModel, x) = f(m, x)

Flux.activations(::typeof(identity), x::Array{Float32,2}) = (x,)

# Base.hash(m::ArrayModel{T}, h::UInt) where {T} = hash((T, m.m), h)
# (m1::ArrayModel{T} == m2::ArrayModel{T}) where {T} = m1.m == m2.m
