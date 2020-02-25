import Base: iterate
using Base: SizeUnknown, EltypeUnknown

struct NodeIterator{T}
    t::T
end

struct LeafIterator{T}
    t::T
end

struct TypeIterator{T, U}
    t::U
    TypeIterator{T}(t::U) where {T, U} = new{T, U}(t)
end

struct PredicateIterator{T}
    t::T
    f::Function
end

struct ZipIterator
    its::Tuple
end

ZipIterator(its...) = ZipIterator(its)

const IteratorTypes = Union{NodeIterator, LeafIterator, PredicateIterator, TypeIterator}

Base.IteratorSize(::IteratorTypes) = SizeUnknown()
Base.IteratorEltype(::IteratorTypes) = EltypeUnknown()
Base.IteratorSize(::ZipIterator) = SizeUnknown()
Base.IteratorEltype(::ZipIterator) = EltypeUnknown()

function iterate(it::ZipIterator) 
    r = collect(map(iterate, it.its))
    any(isnothing.(r)) && return nothing
    return collect(zip(r...))
end

function iterate(it::ZipIterator, ss)
    r = [iterate(i, s) for (i, s) in zip(it.its, ss)]
    any(isnothing.(r)) && return nothing
    return collect(zip(r...))
end

function iterate(it::T, s=Any[it.t]) where T <: IteratorTypes
    r = nextstate(it, s) 
    isnothing(r) && return nothing
    return r, s
end

expand(n, s) = append!(s, reverse(collect(children(n))))

function nextstate(it, s)
    isempty(s) && return nothing
    n = pop!(s)
    processnode(it, n, s)
end

# TODO named tuples?
function processnode(it::NodeIterator, n, s)
    expand(n, s)
    n
end

processnode(it::LeafIterator, n::T, s) where T = processnode(NodeType(T), it, n, s)
processnode(::LeafNode, it, n, s) = n
processnode(::InnerNode, it, n, s) = nextstate(it, expand(n, s))

function processnode(it::PredicateIterator, n, s)
    expand(n, s)
    it.f(n) ? n : nextstate(it, s)
end

function processnode(it::TypeIterator{T, U}, n::T, s) where {T, U}
    expand(n, s)
    n
end
processnode(it::TypeIterator, n, s) = nextstate(it, expand(n, s))
