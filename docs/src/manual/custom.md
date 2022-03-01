```@setup custom
using Mill
using Flux
```

## Custom nodes

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) data nodes are lightweight wrappers around data, such as `Array`, `DataFrame`, and others. It is of course possible to define a custom data (and model) nodes. A useful abstraction for implementing custom data nodes suitable for most cases  is [`LazyNode`](@ref), which you can easily use to extend the functionality of `Mill`.

### Unix path example

Let's define a custom node type for representing path names in Unix and one custom model type for processing it. [`LazyNode`](@ref)
serves as a bolierplate for simple extension of `Mill` ecosystem. We start by by defining an example of such node:

```@repl custom
ds = LazyNode{:Path}(["/var/lib/blob_files/myfile.blob"])
```

Entirely new type is not needed, because we can dispatch on the first type parameter. Specifically,
`:Path` "tag" in this case defines a special kind of [`LazyNode`](@ref). Consequently, we can define
multiple variations of custom [`LazyNode`](@ref) without any conflicts in dispatch.

As a next step, we extend the [`Mill.unpack2mill`](@ref) function, which always takes one [`LazyNode`](@ref)
and produces an arbitrary `Mill` structure. We will represent individual file and directory names (as obtained
by `splitpath`) using an [`NGramMatrix`](@ref) representation and, for simplicity, the whole path as
a bag of individual names:

```@example custom
function Mill.unpack2mill(ds::LazyNode{:Path})
    ss = splitpath.(ds.data)
    x = NGramMatrix(reduce(vcat, ss), 3)
    BagNode(ArrayNode(x), Mill.length2bags(length.(ss)))
end
```
```@repl custom
Mill.unpack2mill(ds)
```

Also, note that the node keeps an array of strings instead of just one string. This is because we
want our node to be able to hold multiple observations than one. Methods such as [`catobs`](@ref)
work as expected:

```@example custom
ds1 = LazyNode{:Path}(["/var/lib/blob_files/myfile.blob"])
ds2 = LazyNode{:Path}(["/var/lib/python"])
nothing # hide
```

```@repl custom
ds = catobs(ds1, ds2)
```

The [`Mill.unpack2mill`](@ref) function is called lazily during the inference by a [`LazyModel`](@ref) counterpart.

[Model reflection](@ref) works too:

```@repl custom
pm = reflectinmodel(ds, d -> Dense(d, 3))
```

We can use the obtained model to perform inference as we would do with any other model.

```@repl custom
pm(ds)
```

### Adding custom nodes without [`LazyNode`](@ref)

The solution using [`LazyNode`](@ref) is sufficient in most scenarios. For other cases, it is recommended to equip custom nodes with the following functionality:

* allow nesting (if needed)
* implement [`Mill.subset`](@ref) and optionally `Base.getindex` to obtain subsets of observations. `Mill` already defines [`Mill.subset`](@ref) for common datatypes, which can be used.
* allow concatenation of nodes with [`catobs`](@ref). Optionally, implement `reduce(catobs, ...)` as well to avoid excessive compilations if a number of arguments will vary a lot
* define a specialized method for `StatsBase.nobs`
* register the custom node with [HierarchicalUtils.jl](@ref) to obtain pretty printing, iterators and other functionality

Here is an example of a custom node with the same functionality as in the [Unix path example](@ref)
section:

```@example custom
using Mill

import Base: getindex, show
import Mill: catobs, data, metadata, VecOrRange, AbstractMillNode, reflectinmodel
import Flux
import StatsBase: nobs
import HierarchicalUtils: NodeType, LeafNode

struct PathNode{S <: AbstractString, C} <: AbstractMillNode
    data::Vector{S}
    metadata::C
end

PathNode(data::Vector{S}) where {S <: AbstractString} = PathNode(data, nothing)
Base.show(io::IO, n::PathNode) = print(io, "PathNode ($(nobs(n)) obs)")

Base.ndims(n::PathNode) = Colon()
nobs(n::PathNode) = length(n.data)
catobs(ns::PathNode) = PathNode(vcat(data.(ns)...), catobs(metadata.(as)...))
Base.getindex(n::PathNode, i::VecOrRange{<:Int}) = PathNode(subset(data(x), i),
                                                            subset(metadata(x), i))
NodeType(::Type{<:PathNode}) = LeafNode()
nothing # hide
```
We also have to define a corresponding model node type which will be a counterpart processing the data:

The solution using [`LazyNode`](@ref) is sufficient in most scenarios. For other cases, it is recommended to equip custom nodes with the following functionality:

Flux.@functor PathModel
show(io::IO, n::PathModel) = print(io, "PathModel")
NodeType(::Type{<:PathModel}) = LeafNode()

path2mill(ds::PathNode) = path2mill(ds.data)
path2mill(ss::Vector{<:AbstractString}) = reduce(catobs, map(path2mill, ss))
function path2mill(s::String)
    ss = splitpath(s)
    BagNode(ArrayNode(NGramMatrix(ss, 3)), AlignedBags([1:length(ss)]))
end

(m::PathModel)(x::PathNode) = m.m(m.path2mill(x))

function reflectinmodel(ds::PathNode, args...)
    pm = reflectinmodel(path2mill(ds), args...)
    PathModel(pm, path2mill)
end
nothing # hide
```

Example of usage:

```@repl custom
ds = PathNode(["/etc/passwd", "/home/tonda/.bashrc"])
pm = reflectinmodel(ds, d -> Dense(d, 3))
pm(ds).data
```
