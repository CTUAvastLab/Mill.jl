struct LazyNode{Name,D} <: AbstractNode
	data::D
end

LazyNode(Name::Symbol, values::T) where {T} = LazyNode{Name,T}(values)
LazyNode{Name}(values::T) where {Name, T} = LazyNode{Name,T}(values)

Base.ndims(x::LazyNode) = Colon()
StatsBase.nobs(a::LazyNode) = length(a.data)
StatsBase.nobs(a::LazyNode, ::Type{ObsDim.Last}) = nobs(a.data)

function Base.reduce(::typeof(Mill.catobs), as::Vector{LazyNode{N,M}}) where {N, M}
    data = reduce(vcat, [x.data for x in as])
    LazyNode{N}(data)
end

Base.getindex(x::LazyNode{N,T}, i::VecOrRange{<:Int}) where {N,T} = LazyNode{N}(subset(x.data, i))

Base.hash(e::LazyNode, h::UInt) = hash((e.data), h)
(e1::LazyNode == e2::LazyNode) = e1.data == e2.data
Base.isequal(e1::LazyNode, e2::LazyNode) = isequal(e1.data, e2.data)
