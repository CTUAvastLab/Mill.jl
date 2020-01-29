struct TreeNode{T,C} <: AbstractTreeNode
    data::T
    metadata::C

    function TreeNode{T,C}(data::T, metadata::C) where {T, C}
        @assert length(data) >= 1 && all(x -> nobs(x) == nobs(data[1]), data)
        new(data, metadata)
    end
end

TreeNode(data::T) where {T} = TreeNode{T, Nothing}(data, nothing)
TreeNode(data::T, metadata::C) where {T, C} = TreeNode{T, C}(data, metadata)

Flux.@functor TreeNode

mapdata(f, x::TreeNode) = TreeNode(map(i -> mapdata(f, i), x.data), x.metadata)

Base.ndims(x::AbstractTreeNode) = Colon()
LearnBase.nobs(a::AbstractTreeNode) = nobs(a.data[1], ObsDim.Last)
LearnBase.nobs(a::AbstractTreeNode, ::Type{ObsDim.Last}) = nobs(a)

function reduce(::typeof(catobs), as::Vector{T}) where {T <: TreeNode}
    data = _cattrees([x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    TreeNode(data, metadata)
end

Base.getindex(x::TreeNode, i::VecOrRange) = TreeNode(subset(x.data, i), subset(x.metadata, i))

function dsprint(io::IO, n::AbstractTreeNode; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "TreeNode$(tr_repr(s, tr))", color=c)
    m = length(n.data)
    ks = key_labels(n.data)
    for i in 1:(m-1)
        println(io)
        paddedprint(io, "  ├── $(ks[i])", color=c, pad=pad)
        dsprint(io, n.data[i], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(ks[i]))))], s=s * encode(i, m), tr=tr)
    end
    println(io)
    paddedprint(io, "  └── $(ks[end])", color=c, pad=pad)
    dsprint(io, n.data[end], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(ks[end]))))], s=s * encode(m, m), tr=tr)
end

