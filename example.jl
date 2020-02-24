# TODO pairs iterator pres schema + data

import Mill.HierarchicalUtils: print_tree, head_string, children_string, children, nchildren, NodeType, LeafNode, InnerNode

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

macro infix(expr) parseinfix(expr) end
parseinfix(e::Expr) = Operation(eval(e.args[1]), collect(map(parseinfix, e.args[2:end])))
parseinfix(x::Number) = Value(x)

evalinfix(n::Value) = n.x
evalinfix(n::Operation) = n.op(evalinfix.(n.ch)...)

t1 = @infix ((2 + 3) * 5) / (4 - 1)
t2 = @infix 1 + 1/0
# TODO indexing?
t2.ch[2].ch[2] = t2

# PRINTING
print_tree(t1)
print_tree(t2; trunc_level=10)
print_tree(t1; trav=true)
