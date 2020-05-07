"""
	struct MissingNode{D} <:AbstractNode
		data::D
		present::Vector{Bool}
	end

	Missing node adds a generic support for missing values to nodes. It wraps the node,
	such that `data` field contain only non-missing values, which are indicated in the
	`present` mask.

"""
struct MissingNode{D} <:AbstractNode
	data::D
	present::Vector{Bool}
end

MissingNode(d) = MissingNode(d, fill(true, nobs(d)))

Base.ndims(x::MissingNode) = Colon()
LearnBase.nobs(a::MissingNode) = length(a.present)
LearnBase.nobs(a::MissingNode, ::Type{ObsDim.Last}) = nobs(a.present)

function Base.reduce(::typeof(Mill.catobs), as::Vector{T}) where {T<:MissingNode}
    data = reduce(Mill.catobs, [x.data for x in as])
    present = reduce(vcat, [x.present for x in as])
    MissingNode(data, present)
end

get_present_index(present, i::Int) = sum(view(present, 1:i))
get_present_index(present, ii::Vector{Int}) = map(i -> get_present_index(present, i), ii)

function Base.getindex(x::MissingNode, i::VecOrRange)
	p = x.present[i]
	!any(p) && return(MissingNode(x.data[1:0], p))
	ii = get_present_index(x.present, i[p])
	@show typeof(p)
	MissingNode(x.data[ii], p)
end

NodeType(::Type{<:MissingNode}) = SingletonNode()
children(n::MissingNode) = (n.data,)
noderepr(n::MissingNode) = "Missing"
childrenfields(::Type{MissingNode}) = (:data,)

Base.hash(e::MissingNode{D}, h::UInt) where {D} = hash((D, e.data, e.present), h)
Base.:(==)(e1::MissingNode{D}, e2::MissingNode{D}) where {D} = e1.data == e2.data && e1.present == e2.present
Base.:(==)(e1::MissingNode{<:Any}, e2::MissingNode{<:Any}) = false
