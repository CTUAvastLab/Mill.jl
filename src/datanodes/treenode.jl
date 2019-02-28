mutable struct TreeNode{T,C} <: AbstractTreeNode{T, C}
    data::T
    metadata::C

    function TreeNode{T,C}(data::T, metadata::C) where {T <: NTuple{N, AbstractNode} where N, C}
        @assert length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data)
        new(data, metadata)
    end
end

TreeNode(data::T) where {T} = TreeNode{T, Nothing}(data, nothing)
TreeNode(data::T, metadata::C) where {T, C} = TreeNode{T, C}(data, metadata)

mapdata(f, x::TreeNode) = TreeNode(map(i -> mapdata(f, i), x.data), x.metadata)

Base.ndims(x::TreeNode) = 0
LearnBase.nobs(a::AbstractTreeNode) = nobs(a.data[1], ObsDim.Last)
LearnBase.nobs(a::AbstractTreeNode, ::Type{ObsDim.Last}) = nobs(a)

function reduce(::typeof(catobs), as::Vector{T}) where {T<:TreeNode}
    data = _cattuples([x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    TreeNode(data, metadata)
end

Base.getindex(x::TreeNode, i::VecOrRange) = TreeNode(subset(x.data, i), subset(x.metadata, i))

function dsprint(io::IO, n::AbstractTreeNode; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "TreeNode$(tr_repr(s, tr))\n", color=c)

    m = length(n.data)
    for i in 1:(m-1)
        paddedprint(io, "  ├── ", color=c, pad=pad)
        dsprint(io, n.data[i], pad=[pad; (c, "  │   ")], s=s * encode(i, m), tr=tr)
    end
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.data[end], pad=[pad; (c, "      ")], s=s * encode(m, m), tr=tr)
end
