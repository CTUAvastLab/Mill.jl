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

find_struct(p, m::AbstractArray{<:Number}, path=[]) = return p === m ? path : nothing
function find_struct(p, m, path=[])
    for (k,n) in Flux.trainable(m) |> pairs
        fs = find_struct(p, n, vcat(path, k))
        isnothing(fs) || return fs
    end
end

# Base.hash(m::MissingModel{T,V}, h::UInt) where {T,V} = hash((T, V, m.m, m.θ), h)
# (m1::MissingModel{T,V} == m2::MissingModel{T,V}) where {T,V} = m1.m == m2.m && m1.θ == m2.θ
