module HierarchicalUtils

const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]


export print

head_string(::T) where T = @error "Define head_string(x) for type $T of x for hierarchical printing, empty string is possible"
tail_string(::T) where T = @error "Define tail_string(x) for type $T of x for hierarchical printing, empty string is possible"
children_string(::T) where T = @error "Define children_string(x) for type $T of x returning an iterable of descriptions for each child, empty strings are possible"
children(::T) where T = @error "Define children(x) for type $T of x returning an iterable of children of x"
nchildren(x) = length(children)
# TODO

abstract type NodeType end
struct LeafNode <: NodeType end
struct InnerNode <: NodeType end
NodeType(::Type{T}) where T = @error "Define NodeType(::Type{$T}) to be either LeafNode() or InnerNode()"

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

# TODO default print Base.show zavolat truncated _print pro ty typy, co na to definuji trait
# TODO listy jsou ti, co nemaji zadne deti
# TODO v Millu zadefinovat Base.show jako print
function _print(io::IO, n::T, C, d, p, e, trav, trunc) where T
    c = NodeType(T) == LeafNode() ? :white : C[1+d%length(C)]
    paddedprint(io, head_string(n) * (trav ? ' ' * "[\"$(stringify(e))\"]" : "") * '\n', color=c)
    CH, CHS = children(n), children_string(n)
    nch = nchildren(n)
    for (i, (ch, chs)) in enumerate(zip(CH, CHS))
        paddedprint(io, "  " * (i == nch ? "└" : "├") * "── " * chs, color=c, pad=p)
        ns = (i == nch ? "   " : "  │") * repeat(" ", max(3, 2+length(chs)))
        _print(io, ch, C, d+1, [p; (c, ns)], e * encode(i, nch), trav, trunc)
    end
    paddedprint(io, tail_string(n), color=c)
end

include("traversal_encoding.jl")

end # module
