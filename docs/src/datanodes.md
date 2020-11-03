# Architecture of Data nodes

DataNodes are lightweight wrappers around data, such as Array, DataFrames, etc. Their primary purpose is to allow their nesting. It is possible to create subsets using `getindex` and concatenate them using `cat`. Note that internally, nodes are concatenated using `reduce(catobs, ...)`. Similarly `getindex` is redirected to `subset`. This is needed such that operations over `Matrices`, `Vectors`, and `DataFrames` are consistent with the library.

Let's take a look on an example on a simple `ArrayNode` holding a matrix or a vectors.

`ArrayNode` has a simple structure holding only Array, which is considered the data and optionally some metadata, which can be literally anything.
```
struct ArrayNode{A,C} <: AbstractNode
    data::A
    metadata::C
end
```


`ArrayNode` had overloaded a `getindex` to support indexing. But the `getindex` just calls `subset(x::ArrayNode, idxs)`, which is used to correctly slice arrays according to the last dimension.

**This mean that if you want to define your own DataNode, in order to be compatible with the rest of the library it has to implement `subset` and `reduce(::typeof{catobs}, Vector{T}) where {T<:YourType}`**



# An Example of how to add a custom datatype --- a container for unix pathnames struct
We give it a twist, such that the extractor will be part of the model definition, which is going to be cute.

Let's start by defining the structure holding pathnames, supporting `nobs` joining of two structures and indexing into the structure. A last touch is to extend the pretty printing.
```
struct PathNode{S<:AbstractString,C} <: AbstractNode
    data::Vector{S}
    metadata::C
end

PathNode(data::Vector{S}) where {S<:AbstractString} = PathNode(data, nothing)

Base.ndims(x::PathNode) = Colon()
StatsBase.nobs(a::PathNode) = length(a.data)
StatsBase.nobs(a::PathNode, ::Type{ObsDim.Last}) = nobs(a)

function Base.reduce(::typeof(Mill.catobs), as::Vector{T}) where {T<:PathNode}
    data = reduce(vcat, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    PathNode(data, metadata)
end

Base.getindex(x::PathNode, i::VecOrRange) = PathNode(subset(x.data, i), subset(x.metadata, i))

Mill.dsprint(io::IO, n::PathNode; pad=[], s="", tr=false) = paddedprint(io, "PathNode$(size(n.data))$(tr_repr(s, tr))")
```

Similarly, we define a `ModelNode` which will be a counterpart processing the data. Note that the part of the `ModelNode` is a function which converts the pathanme string to `Matrix` (or other Mill structures). Again, we add a support for pretty printing.

```
struct PathModel{T,F} <: AbstractMillModel
    m::T
    path2mill::F
end

Flux.@functor PathModel

(m::PathModel)(x::PathNode)  = m.m(m.path2mill(x))

function Mill.modelprint(io::IO, m::PathModel; pad=[], s="", tr=false)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "PathModel$(tr_repr(s, tr))\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    modelprint(io, m.m, pad=[pad; (c, "      ")])
end
```

Finally, let's define function path2mill, which converts
a list of strings to Mill internal structure.
```
function path2mill(s::String)
	ss = String.(split(s, "/"))
	BagNode(ArrayNode(Mill.NGramMatrix(ss, 3, 256, 2053)), AlignedBags([1:length(ss)]))
end

path2mill(ss::Vector{S}) where {S<:AbstractString} = reduce(catobs, map(path2mill, ss))
path2mill(ds::PathNode) = path2mill(ds.data)

```

And then, let's test the solution

```
ds = PathNode(["/etc/passwd", "/home/tonda/.bashrc"])
pm = PathModel(reflectinmodel(path2mill(ds), d -> Dense(d, 10, relu)), path2mill)
pm(ds).data
```

A final touch would be to overload the `reflectinmodel` as
```

function Mill.reflectinmodel(ds::PathNode, args...)
	pm = reflectinmodel(path2mill(ds), args...)
	PathModel(pm, path2mill)
end

```
which can make it seamless
```
ds = PathNode(["/etc/passwd", "/home/tonda/.bashrc"])
pm = reflectinmodel(ds, d -> Dense(d, 10, relu))
pm(ds).data
```
