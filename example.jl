import Mill.HierarchicalUtils: print_tree, head_string, children_string, children, nchildren, NodeType, LeafNode, InnerNode, NodeIterator, LeafIterator, PredicateIterator, TypeIterator, ZipIterator, nnodes, nleafs, encode_traversal, walk

import Base.getindex

abstract type Expression end

struct Value <: Expression
    x::Number
end

NodeType(::Type{Value}) = LeafNode()
head_string(n::Value) = string(n.x)

struct Operation <: Expression
    op::Function
    ch::Vector{Expression}
end

NodeType(::Type{Operation}) = InnerNode()
children(n::Operation) = n.ch
head_string(n::Operation) = string(n.op)

# TODO delete
# Base.show(io::IO, ::MIME"text/plain", n::T) where T = print(io, Base.typename(T))
# Base.show(io::IO, n::T) where T = print(io, Base.typename(T))

macro infix(expr) parseinfix(expr) end
parseinfix(e::Expr) = Operation(eval(e.args[1]), collect(map(parseinfix, e.args[2:end])))
parseinfix(x::Number) = Value(x)

evalinfix(n::Value) = n.x
evalinfix(n::Operation) = n.op(evalinfix.(n.ch)...)

t1 = @infix ((2 + 3) * 5) / (4 - 1)

# PRINTING
print_tree(t1)

# TRAVERSALS
Base.getindex(t::Operation, i::Integer) = t.ch[i]
Base.getindex(t::Operation, idxs::NTuple{N, Integer}) where N = t.ch[idxs[1]][idxs[2:end]...]
Base.getindex(t::Operation, idxs::Integer...) = t[idxs]
Base.getindex(t::Expression, i::AbstractString) = walk(t, i)
print_tree(t1; trav=true)
encode_traversal(t1, 1, 1)

t1[1,1] === t1[1][1] == t1[encode_traversal(t1, 1, 1)]

# TRUNCATION
print_tree(t2; trunc_level=10)
t2 = @infix 1 + 1/0
t2.ch[2].ch[2] = t2

# TODO ukazat uz na Millu?
collect(NodeIterator(t1))
collect(LeafIterator(t1))
collect(TypeIterator{Value}(t1))
collect(TypeIterator{Operation}(t1))
collect(TypeIterator{Expression}(t1))

pred(n::Operation) = in(n.op, [+, -])
pred(n::Value) = isodd(n.x)
collect(PredicateIterator(t1, pred))

collect(ZipIterator(NodeIterator(t1), NodeIterator(t1)))
collect(ZipIterator(NodeIterator(t1), LeafIterator(t1)))

nnodes(t1)
nleafs(t1)
