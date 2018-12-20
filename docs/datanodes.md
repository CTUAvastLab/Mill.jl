# Architecture of Data nodes

DataNodes are lightweight wrappers around data, such as Array, DataFrames, etc. Their primary purpose is to allow their nesting. It is possible to create subsets using `getindex` and concatenate them using `cat`. Note that internally, nodes are concatenated using `reduce(catobs, ...)`. Similarly `getindex` is redirected to `subset`. This is needed such that operations over `Matrices`, `Vectors`, and `DataFrames` are consistent with the library.

Let's take a look on an example on a simple `ArrayNode` holding a matrix or a vectors.

`ArrayNode` has a simple structure holding only Array, which is considered the data and optionally some metadata, which can be literally anything.
```
mutable struct ArrayNode{A,C} <: AbstractNode
    data::A
    metadata::C
end
```


`ArrayNode` had overloaded a `getindex` to support indexing. But the `getindex` just calls `subset(x::ArrayNode, idxs)`, which is used to correctly slice arrays according to the last dimension. 

**This mean that if you want to define your own DataNode, in order to be compatible with the rest of the library it has to implement `subset` and `reduce(::typeof{catobs}, Vector{T}) where {T<:YourType}**