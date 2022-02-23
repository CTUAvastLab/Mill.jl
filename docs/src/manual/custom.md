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

### Unix path example

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
pm(ds).data
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

```@example custom
struct PathModel{T, F} <: AbstractMillModel
    m::T
    path2mill::F
end

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
