```@setup custom
using Mill
using Flux
```

## Custom nodes

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) data nodes are lightweight wrappers around data, such as 
`Array`, `DataFrame`, and others. When implementing custom nodes, it is recommended to equip them with the following 
functionality to fit better into [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) environment:

* allow nesting (if needed)
* implement `getindex` to obtain subsets of observations. For this purpose, [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) defines a [`Mill.subset`](@ref) function for common datatypes, which can be used.
* allow concatenation of nodes with [`catobs`](@ref). Optionally, implement `reduce(catobs, ...)` as well to avoid excessive compilations if a number of arguments will vary a lot
* define a specialized method for `nobs`
* register the custom node with [HierarchicalUtils.jl](@ref) to obtain pretty printing, iterators and other functionality

luckily, for most cases you don't need to implement this all by yourself, and you can use [`LazyNode`](@ref) which implements mose of this logic, so you can extend the functionality of Mill easily.
The following example shows how to use the [`LazyNode`](@ref) in order to make a custom node.

### Unix path example

Let's use the [`LazyNode`](@ref) as a custom node type for representing path names in Unix and define a function to parse 
the path in string form in to the [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) structure. 

The [`LazyNode`](@ref) lets us wrap the data (path as a string) into out custom parametric type, which is expanded lazily during the inference (hence the name).
```@example custom
ds = LazyNode{:Path}(["/var/lib/blob_files/myfile.blob"])
```
it's important to have the `{:Path}` parameter there, because it's used to dispatch to the `Mill.unpack2mill(ds::LazyNode{:Path})`, which we'll define below.

Note that the node keeps array of strings. That's because the `LazyNode` will always keep the array of strings,
because internally the structure is designed to hold minibatch of any size, so 1 sample is batch of size 1,
but we can merge multiple samples together, and it will still produce sample with array of strings.
See
```@example custom
ds2 = LazyNode{:Path}(["/var/lib/python"])
ds3 = reduce(catobs, [ds, ds2])
```

Then we can define the function which takes the [`LazyNode`](@ref)`{:Path}` containing data as a vector of strings and produces a [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) structure.
For simplicity, we will represent the path as a bag of individual directory and file names, which we obtain by splitting the path by `/`. 
One could hypothesize that using ordered sequences would provide more information because we would not lose the ordering, but
let's keep it simple for now. We will represent individual directory names using a [`NGramMatrix`](@ref) representation in this example.

```@example custom
function Mill.unpack2mill(ds::LazyNode{:Path})
    ss = split.(ds.data, "/")
    x = NGramMatrix(reduce(vcat, ss), 3)
    BagNode(ArrayNode(x), Mill.length2bags(length.(ss)))
end
```

Internally, [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) calls the `unpack2mill` function, but we can try to call it explicitly to see if it works.
```@example custom
Mill.unpack2mill(ds)
```

Lastly, we can create the model using the common functionality.
```@repl custom
pm = reflectinmodel(ds)
```

this creates [`LazyModel`](@ref), which is able to work with [`LazyNode`](@ref)s.

Then we can use it to perform inference as we would do with any other model.

```@repl custom
pm(ds).data
```

### Adding custom nodes which don't fit into the [`LazyNode`](@ref) formalism

Of course the [`LazyNode`](@ref) might not cover all your needs, in which case see the below example how to plug in any functionality to Mill without the usage of LazyNode, 
so here is the checklist of what you should implement if you want the functionality to fit into [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) environment nicely: 

* allow nesting (if needed)
* implement `getindex` to obtain subsets of observations. For this purpose, `Mill` defines a [`Mill.subset`](@ref) function for common datatypes, which can be used.
* allow concatenation of nodes with [`catobs`](@ref). Optionally, implement `reduce(catobs, ...)` as well to avoid excessive compilations if a number of arguments will vary a lot
* define a specialized method for `nobs`
* register the custom node with [HierarchicalUtils.jl](@ref) to obtain pretty printing, iterators and other functionality

#### Unix path example

Let's define one custom node type for representing pathnames in Unix and one custom model type for processing it. We'll start by defining the structure holding pathnames:

```@example custom
struct PathNode{S <: AbstractString, C} <: AbstractMillNode
    data::Vector{S}
    metadata::C
end

PathNode(data::Vector{S}) where {S <: AbstractString} = PathNode(data, nothing)

Base.show(io::IO, n::PathNode) = print(io, "PathNode ($(nobs(n)) obs)")
nothing # hide
```

We will support `nobs`:

```@example custom
import StatsBase: nobs
Base.ndims(x::PathNode) = Colon()
nobs(a::PathNode) = length(a.data)
nothing # hide
```

concatenation:

```@example custom
function Base.reduce(::typeof(catobs), as::Vector{T}) where {T <: PathNode}
    PathNode(data, reduce(vcat, data.(as)), reduce(catobs, metadata.(as)))
end
```

and indexing:

```@example custom
function Base.getindex(x::PathNode, i::Mill.VecOrRange{<:Int})
    PathNode(Mill.subset(Mill.data(x), i), Mill.subset(Mill.metadata(x), i))
end
```

The last touch is to add the traits needed by [HierarchicalUtils.jl](@ref):

```@example custom
import HierarchicalUtils
HierarchicalUtils.NodeType(::Type{<:PathNode}) = HierarchicalUtils.LeafNode()
nothing # hide
```

Now, we are ready to create the first `PathNode`:

```@repl custom
ds = PathNode(["/etc/passwd", "/home/tonda/.bashrc"])
```

Similarly, we define a model node type which will be a counterpart processing the data:

```@example custom
struct PathModel{T, F} <: AbstractMillModel
    m::T
    path2mill::F
end

Base.show(io::IO, n::PathModel) = print(io, "PathModel")
Flux.@functor PathModel
```

Note that the part of the model node is a function which converts the pathname string to a `Mill` structure. For simplicity, we use a trivial [`NGramMatrix`](@ref) representation in this example and define `path2mill` as follows:

```@example custom
```

Now we define how the model node is applied:

```@example custom
(m::PathModel)(x::PathNode) = m.m(m.path2mill(x))
```

And again, define everything needed in [HierarchicalUtils.jl](@ref):

```@example custom
HierarchicalUtils.NodeType(::Type{<:PathModel}) = HierarchicalUtils.LeafNode()
```

Let's test that everything works:

```@repl custom
pm = PathModel(reflectinmodel(path2mill(ds)), path2mill)
pm(ds).data
```

The final touch would be to overload the [`reflectinmodel`](@ref) as

```@example custom
function Mill.reflectinmodel(ds::PathNode, args...)
    pm = reflectinmodel(path2mill(ds), args...)
    PathModel(pm, path2mill)
end
```

which makes things even easier

```@repl custom
pm = reflectinmodel(ds)
pm(ds).data
```
