# TODO
# Iterators - type of node, 
# indexing - do Millu?
# treemap, treemap!, map na listy a vratit seznam vysledku?
# reflectinmodel -> treemap
# map over multiple trees with the same structure simultaneously using a function of multiple arguments.
# Work with individual samples and not whole batches. 
# Matej neco na ten zpusob co chtel
# tests
# nodecount
# leafcount
# pairs iterator
# mutating?
# odebrat vse z Millu, zmenit dokumentaci Millu

module HierarchicalUtils

const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

abstract type NodeType end
struct LeafNode <: NodeType end
struct InnerNode <: NodeType end
NodeType(::Type{T}) where T = @error "Define NodeType(::Type{$T}) to be either LeafNode() or InnerNode()"

children_string(x::T) where T = children_string(NodeType(T), x)
children_string(::LeafNode, _) = []
# children_string(::InnerNode, ::T) where T = @error "Define children_string(x) for type $T of x returning an iterable of descriptions for each child, empty strings are possible"
children_string(::InnerNode, n::T) where T = ["" for _ in 1:length(children(n))]

children(x::T) where T = children(NodeType(T), x)
children(::InnerNode, ::T) where T = @error "Define children(x) for type $T of x returning an iterable of children of x"
children(::LeafNode, _) = []

nchildren(x::T) where T = nchildren(NodeType(T), x)
nchildren(::LeafNode, _) = 0
nchildren(::InnerNode, x) = length(children(x))

head_string(::T) where T = @error "Define head_string(x) for type $T of x for hierarchical printing, empty string is possible"
# tail_string(::InnerNode, ::T) where T = @error "Define tail_string(x) for type $T of x for hierarchical printing, empty string is possible"

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

function _print_tree(io::IO, n::T, C, d, p, e, trav, trunc_level) where T
    c = NodeType(T) == LeafNode() ? :white : C[1+d%length(C)]
    gap = " " ^ min(2, length(head_string(n))-1)
    paddedprint(io, head_string(n) * (trav ? ' ' * "[\"$(stringify(e))\"]" : ""), color=c)
    nch = nchildren(n)
    if nch > 0 && d >= trunc_level
        println(io)
        paddedprint(io, gap * '⋮', color=c, pad=p)
    elseif nch > 0
        CH, CHS = children(n), children_string(n)
        for (i, (ch, chs)) in enumerate(zip(CH, CHS))
            println(io)
            paddedprint(io, gap * (i == nch ? "└" : "├") * "── " * chs, color=c, pad=p)
            ns = gap * (i == nch ? ' ' : '│') * repeat(" ", max(3, 2+length(chs)))
            _print_tree(io, ch, C, d+1, [p; (c, ns)], e * encode(i, nch), trav, trunc_level)
        end
    end
    # paddedprint(io, tail_string(n), color=c)
end

print_tree(n::T; trav=false, trunc_level=Inf) where T = print_tree(stdout, n, trav=trav, trunc_level=trunc_level)
print_tree(io::IO, n::T; trav=false, trunc_level=Inf) where T = _print_tree(io, n, COLORS, 0, [], "", trav, trunc_level)

# TODO imports
include("iterators.jl")
include("traversal_encoding.jl")

# TODO exports
export print


end # module
