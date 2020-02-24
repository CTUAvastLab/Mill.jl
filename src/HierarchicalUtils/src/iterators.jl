import Base: iterate
using Base: SizeUnknown, EltypeUnknown

struct LeafIterator{T}
    t::T
end

struct PredicateIterator{T}
    t::T
    f::Function
end

# TODO jeden velky abstraktni typ?
# TODO moznost otypovat???
Base.IteratorSize(::Union{LeafIterator, PredicateIterator}) = SizeUnknown()
Base.IteratorEltype(::Union{LeafIterator, PredicateIterator}) = EltypeUnknown()

function iterate(it::T, s=Any[it.t]) where T <: Union{LeafIterator, PredicateIterator}
    r = nextstate(it, s) 
    isnothing(r) && return nothing
    return r, s
end

expand(it, n, s) = nextstate(it, append!(s, reverse(collect(children(n)))))

# TODO fix bug in predicate, when predicate is true, we want to continue!
function nextstate(it, s)
    isempty(s) && return nothing
    n = pop!(s)
    processnode(it, n, s)
end

# TODO prochazet dva stromy se stejnou strukturou
# TODO named tuples?

function processnode(it::PredicateIterator, n, s)
    append!(s, reverse(collect(children(n))))
    it.f(n) ? n : nextstate(it, s)
end

processnode(::LeafIterator, it, n::T, s) where T = processnode(NodeType(T), it, n, s)
processnode(::LeafNode, it, n, s) = n
processnode(::InnerNode, it, n, s) = expand(it, n, s)




