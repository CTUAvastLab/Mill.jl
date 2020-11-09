## Directed Acyclic Graphs in Mill.jl

In this exercice, I was interesting in a following problem. Imagine a data / knowlege base represented in a form of a directed acyclic graph (DAG), where the vertex would be modelled on basis of its parents (and their parents), but not on its descendants. I was further interested in a case, where the descendants of some vertex `i` would represent only subset of all nodes in the graph. How to do this with a least pain and efficiently?

In the course of calculating the value of a vertex `i`, there can be vertices which can be used more that one time. This means that we would like to have a cache of already calculated values, which is difficult since `Zygote` does not like `setindex` operation. But, the cache is assigned only once, which means that it can be realized through `Zygote.buffer`. So, here we go.

At first, we initiate the cache as
```julia
initcache(g, k) = [Zygote.Buffer(zeros(Float32, k, 1)) for _ in 1:nv(g)]
```

To get the value of a vertex, we just delegate the question to `cache` as 

```julia
function (model::DagModel)(g::DagGraph, i)
  cache = initcache(g.g, model.odim)
  ArrayNode(getfromcache!(cache, g, model, i))
end
```

which means that the `getfromcache!` will do all the heavy lifting. But that function will just check, if the value in `cache` has been already calculated, or it calculates the value (applying `model` on `millvertex!`) and freezes the calculated item in cache.

```julia
function getfromcache!(cache, g::DagGraph, model::DagModel, i::Int)
  cache[i].freeze && return(copy(cache[i]))
  ds = millvertex!(cache, g, model, i)
  cache[i][:] = model.m(ds).data
  return(copy(cache[i]))
end
```

and what `millvertex!` function does? It just takes the representation of ancestors (from `cache`) and put them together

```julia
function millvertex!(cache, g::DagGraph, model::DagModel, i)
  ProductNode((neighbours = millneighbors!(cache, g, model, i), 
    vertex = vertex_features[i])
  )
end
```

Wait a sec, am I running in circles? Yes, and that is the art of recursion. Below is the complete example I have once written. Note that it is not the most efficient approach to implement this. It would be better to spent a little time with graphs to identify sets of vertices that can be processed in parallel and for which all ancestors are know. But this was a fun little exercise.


```julia
using Flux, Zygote
using LightGraphs, MetaGraphs, Mill, Setfield


struct DagGraph{G<:SimpleDiGraph,T}
  g::G
  vertex_features::T
end

Zygote.@nograd LightGraphs.inneighbors

struct DagModel{M}
  m::M
  odim::Int
end

Flux.@functor DagModel


function (model::DagModel)(g::DagGraph, i)
  cache = initcache(g.g, model.odim)
  ArrayNode(getfromcache!(cache, g, model, i))
end

(model::DagModel)(g::SimpleDiGraph, vertex_features, i) = model(DagGraph(g, vertex_features), i)


initcache(g, k) = [Zygote.Buffer(zeros(Float32, k, 1)) for _ in 1:nv(g)]
Zygote.@nograd initcache

function millvertex!(cache, g::DagGraph, model::DagModel, i)
  ProductNode((neighbours = millneighbors!(cache, g, model, i), 
    vertex = vertex_features[i])
  )
end

function getfromcache!(cache, g::DagGraph, model::DagModel, ii::Vector{Int}) 
  reduce(catobs, [getfromcache!(cache, g, model, i) for i in ii])
end

function getfromcache!(cache, g::DagGraph, model::DagModel, i::Int)
  cache[i].freeze && return(copy(cache[i]))
  ds = millvertex!(cache, g, model, i)
  cache[i][:] = model.m(ds).data
  return(copy(cache[i]))
end

function millneighbors!(cache, g::DagGraph, model::DagModel, ii::Vector{Int})
  isempty(ii) && return(BagNode(missing, [0:-1]))
  xs = [getfromcache!(cache, g, model, i) for i in  ii]
  BagNode(ArrayNode(reduce(catobs, xs)), [1:length(xs)])
end

millneighbors!(cache, g::DagGraph, model::DagModel, i::Int) = millneighbors!(cache, g, model, inneighbors(g.g, i))
```
