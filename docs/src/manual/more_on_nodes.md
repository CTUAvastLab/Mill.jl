```@setup more_on_nodes 
using Mill
```

# More on nodes

## Node nesting 

The main advantage of the [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) library is that it allows to arbitrarily nest and cross-product [`BagModel`](@ref)s, as described in Theorem 5 in [Pevny2019](@cite). In other words, instances themselves may be represented in much more complex way than in the [`BagNode`](@ref) and [`BagModel`](@ref) example.

Let's start the demonstration by nesting two MIL problems. The outer MIL model contains three samples (outer-level bags), whose instances are (inner-level) bags themselves. The first outer-level bag contains one inner-level bag problem with two inner-level instances, the second outer-level bag contains two inner-level bags with total of three inner-level instances, and finally the third outer-level bag contains two inner bags with four instances:

```@repl more_on_nodes
ds = BagNode(BagNode(ArrayNode(randn(4, 10)),
                     [1:2, 3:4, 5:5, 6:7, 8:10]),
             [1:1, 2:3, 4:5])
```

Here is one example of a model, which is appropriate for this hierarchy:

```@example more_on_nodes
using Flux: Dense, Chain, relu
```
```@repl more_on_nodes
m = BagModel(
        BagModel(
            ArrayModel(Dense(4, 3, relu)),
            SegmentedMeanMax(3),
            Dense(6, 3, relu)),
        SegmentedMeanMax(3),
        Chain(Dense(6, 3, relu), Dense(3, 2)))
```

and can be directly applied to obtain a result:

```@repl more_on_nodes
m(ds)
```

Here we again make use of the property that even if each instance is represented with an arbitrarily complex structure, we always obtain a vector representation after applying instance model `im`, regardless of the complexity of `im` and `Mill.data(ds)`:

```@repl more_on_nodes
m.im(Mill.data(ds))
```

In one final example we demonstrate a complex model consisting of all types of nodes introduced so far:

```@repl more_on_nodes
ds = BagNode(ProductNode((BagNode(randn(4, 10),
                                  [1:2, 3:4, 5:5, 6:7, 8:10]),
                          randn(3, 5),
                          BagNode(BagNode(randn(2, 30),
                                          [i:i+1 for i in 1:2:30]),
                                  [1:3, 4:6, 7:9, 10:12, 13:15]),
                          randn(2, 5))),
             [1:1, 2:3, 4:5])
```

As data and model trees tend to be complex, `Mill` limits the printing. To inspect the whole tree, use
`printtree`:

```@repl more_on_nodes
printtree(ds)
```

Instead of defining a model manually, we can also make use of [Model reflection](@ref), another `Mill` functionality, which simplifies model creation:

```@repl more_on_nodes
m = reflectinmodel(ds, d -> Dense(d, 2), SegmentedMean)
m(ds)
```

## Node conveniences

To make the handling of data and model hierarchies easier, `Mill` provides several tools. Let's setup some data:

```@repl more_on_nodes
AN = ArrayNode(Float32.([1 2 3 4; 5 6 7 8]))
AM = reflectinmodel(AN)
BN = BagNode(AN, [1:1, 2:3, 4:4])
BM = reflectinmodel(BN)
PN = ProductNode(a=Float32.([1 2 3; 4 5 6]), b=BN)
PM = reflectinmodel(PN)
```

### Function: `numobs`

`numobs` function from [`MLUtils.jl`](https://github.com/JuliaML/MLUtils.jl) returns a number of samples from the current level point of view. This number usually increases as we go down the tree when [`BagNode`](@ref)s are involved, as each bag may contain more than one instance.

```@repl more_on_nodes
numobs(AN)
numobs(BN)
numobs(PN)
```

### Indexing and Slicing

Indexing in [`Mill`] operates **on the level of observations**:

```@repl more_on_nodes
AN[1]
numobs(ans)
BN[2]
numobs(ans)
PN[3]
numobs(ans)
AN[[1, 4]]
numobs(ans)
BN[1:2]
numobs(ans)
PN[[2, 3]]
numobs(ans)
PN[Int[]]
numobs(ans)
```

This may be useful for creating minibatches and their permutations.

Note that apart from the perhaps apparent recurrent effect, this operation requires other implicit actions, such as properly recomputing bag indices:

```@repl more_on_nodes
BN.bags
BN[[1, 3]].bags
```

### Function: [`catobs`](@ref)

[`catobs`](@ref) function concatenates several samples (datasets) together:

```@repl more_on_nodes
catobs(AN[1], AN[4])
catobs(BN[3], BN[[2, 1]])
catobs(PN[[1, 2]], PN[3:3]) == PN
```

Again, the effect is recurrent and everything is appropriately recomputed:

```@repl more_on_nodes
BN.bags
catobs(BN[3], BN[[1]]).bags
```

This operation is an analogy to what is usually done in the classical setting. If every observation is
represented as a vector of features, each (mini)batch of samples is first concatenated into one
matrix and the whole matrix is run through the neural network using fast matrix multiplication
procedures. The same reasoning applies here, but instead of `Base.cat`, [`catobs`](@ref) is needed.

Equipped with everything mentioned above there are two different ways to construct minibatches from
data. First option, applicable mainly to smaller datasets, is to load all avaiable data into memory,
store it as one big data node containing all observations, and use [Indexing and Slicing](@ref) to
obtain minibatches. Such approach is demonstrated in the [Musk](@ref) example. The other option is
to read all observations into memory separately (or load them on demand) and construct minibatches
with [`catobs`](@ref).

!!! ukn "More tips"
    For more tips for handling datasets and models, see [External tools](@ref HierarchicalUtils.jl).

## Metadata

Each [`AbstractMillNode`](@ref) can also carry arbitrary **metadata** (defaulting to `nothing`).
Metadata is provided upon construction of the node and accessed metadata by [`Mill.metadata`](@ref):

```@repl more_on_nodes
n1 = ArrayNode(randn(2, 2), ["metadata"])
Mill.metadata(n1)
n2 = ProductNode(n1, [1 3; 2 4])
Mill.metadata(n2)
```
