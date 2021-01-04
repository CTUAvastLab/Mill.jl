```@setup mill
using Mill
using Flux
```

## Adding custom nodes

`Mill.jl` data nodes are lightweight wrappers around data, such as `Array`, `DataFrame`, and others. When implementing custom nodes, it is recommended to equip them with the following functionality to fit better into `Mill.jl` environment:

* allow nesting (if needed)
* implement `getindex` to obtain subsets of observations. For this purpose, `Mill.jl` defines a `subset` function for common datatypes, which can be used.
* allow concatenation of nodes with `catobs`. Optionally, implement `reduce(catobs, ...)` as well to avoid excessive compilations if a number of arguments will vary a lot
* define a specialized method for `nobs`
* register the custom node with [`HierarchicalUtils.jl`](@ref) to obtain pretty printing, iterators and other functionality

## Unix path example

Let's define one custom node type for representing pathnames in Unix and one custom model type for processing it. We'll start by defining the structure holding pathnames:

```@example mill
struct PathNode{S <: AbstractString, C} <: AbstractNode
    data::Vector{S}
    metadata::C
end

PathNode(data::Vector{S}) where {S <: AbstractString} = PathNode(data, nothing)
nothing # hide
```

We will support `nobs`:

```@example mill
import StatsBase: nobs
Base.ndims(x::PathNode) = Colon()
nobs(a::PathNode) = length(a.data)
nothing # hide
```

concatenation:

```@example mill
function Base.reduce(::typeof(catobs), as::Vector{T}) where {T <: PathNode}
    data = reduce(vcat, [x.data for x in as])
    metadata = reduce(catobs, [a.metadata for a in as])
    PathNode(data, metadata)
end
```

and indexing:

```@example mill
Base.getindex(x::PathNode, i::Mill.VecOrRange{<:Int}) = PathNode(subset(x.data, i), subset(x.metadata, i))
```

The last touch is to add the definition needed by `HierarchicalUtils.jl`:

```@example mill
import HierarchicalUtils
HierarchicalUtils.NodeType(::Type{<:PathNode}) = HierarchicalUtils.LeafNode()
HierarchicalUtils.noderepr(n::PathNode) = "PathNode ($(nobs(n)) obs)"
nothing # hide
```

Now, we are ready to create the first `PathNode`:

```@repl mill
ds = PathNode(["/etc/passwd", "/home/tonda/.bashrc"])
```

Similarly, we define a `ModelNode` type which will be a counterpart processing the data:

```@example mill
struct PathModel{T, F} <: AbstractMillModel
    m::T
    path2mill::F
end

Flux.@functor PathModel
```

Note that the part of the `ModelNode` is a function which converts the pathname string to a `Mill.jl` structure. For simplicity, we use a trivial `NGramMatrix` representation in this example and define `path2mill` as follows:

```@example mill
function path2mill(s::String)
    ss = String.(split(s, "/"))
    BagNode(ArrayNode(Mill.NGramMatrix(ss, 3)), AlignedBags([1:length(ss)]))
end

path2mill(ss::Vector{S}) where {S <: AbstractString} = reduce(catobs, map(path2mill, ss))
path2mill(ds::PathNode) = path2mill(ds.data)
nothing # hide
```

Now we define how the model node is applied:

```@example mill
(m::PathModel)(x::PathNode) = m.m(m.path2mill(x))
```

And again, define everything needed in `HierarchicalUtils.jl`:

```@example mill
HierarchicalUtils.NodeType(::Type{<:PathModel}) = HierarchicalUtils.LeafNode()
HierarchicalUtils.noderepr(n::PathModel) = "PathModel"
```

Let's test that everything works:

```@repl mill
pm = PathModel(reflectinmodel(path2mill(ds)), path2mill)
pm(ds).data
```

The final touch would be to overload the `reflectinmodel` as

```@example mill
function Mill.reflectinmodel(ds::PathNode, args...)
    pm = reflectinmodel(path2mill(ds), args...)
    PathModel(pm, path2mill)
end
```

which makes things even easier

```@repl mill
pm = reflectinmodel(ds)
pm(ds).data
```
