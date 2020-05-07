"""
struct MissingModel{T,V} <: AbstractMillModel
    m::T
    θ::Vector{V}
end

Implements support for missing values in any nodes. The node works as a wrapper, i.e. the wrapped model `m` processes non-missing items and missing items are replaced by `θ`. `θ` are treated as a parameter of the model, which means it is optimized in the back-propagation.
"""
struct MissingModel{T,V} <: AbstractMillModel
    m::T
    θ::Vector{V}
end

Flux.@functor MissingModel

function (m::MissingModel)(x::MissingNode) 
	xx = m.m(x.data)
	all(x.present) && return(xx)
	ArrayNode(fillmissing(x.present, xx.data, m.θ))
end

function fillmissing(present, x, θ)
	o = similar(x, size(x,1), length(present))
	o[:, present]   = x
	o[:, .!present] .= θ
	o
end

Zygote.@adjoint function fillmissing(present, x, θ)
	fillmissing(present, x, θ), Δ -> (nothing, Δ[:,present], sum(Δ[:,.!present], dims = 2)[:])
end


function _reflectinmodel(x::MissingNode, db, da, b, a, s)
	@show x
    im, d = _reflectinmodel(x.data, db, da, b, a, s)
    θ = zeros(Float32, d)
    MissingModel(im, θ), d
end


NodeType(::Type{<:MissingModel}) = SingletonNode()
noderepr(n::MissingModel) = "Missing"
childrenfields(::Type{MissingModel}) = (:m,)
children(n::MissingModel) = (n.data,)