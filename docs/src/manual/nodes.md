```@setup nodes
using Mill
```

!!! ukw "Tip"
    It is recommended to read the [Motivation](@ref) section first to understand the crucial ideas behind hierarchical multiple instance learning.

# Nodes

`Mill.jl` enables representation of arbitrarily complex tree-like hierarchies and appropriate models for these hierarchies. It defines two core abstract types:

1. [`AbstractNode`](@ref) which stores data on any level of abstraction and its subtypes can be further nested
2. [`AbstractModelNode`](@ref) which helps to define a corresponding model. For each specific implementation of `AbstractNode` we have one or more specific `AbstractModelNode`(s) for processing it.

Below we will go through implementation of `ArrayNode`, `BagNode` and `ProductNode` together with their corresponding models. It is possible to define data and model nodes for more complex behaviors (see [Custom Nodes](@ref)), however, these three core types are already sufficient for a lot of tasks, for instance, representing any `JSON` document and using appropriate models to convert it to a vector represention or classify it (see **TODO**).

## `ArrayNode` and `ArrayModel`

`ArrayNode` thinly wraps an array of features (specifically any subtype of `AbstractArray`):

```@repl nodes
X = Float32.([1 2 3 4; 5 6 7 8])
AN = ArrayNode(X)
```

Data carried by any [`AbstractNode`](@ref) can be accessed with the [`Mill.data`](@ref) function as follows:

```@repl nodes
Mill.data(AN)
```

Similarly, `ArrayModel` wraps any function performing operation over this array. In example below, we wrap a feature matrix `X` and a `Dense` model from [`Flux.jl`](https://fluxml.ai):

```@example nodes
using Flux: Dense
```

```@repl nodes
f = Dense(2, 3)
AM = ArrayModel(f)
```

We can apply the model now with `AM(AN)` to get another `ArrayNode` and verify that the feedforward layer `f` is really applied:

```@repl nodes
AM(AN)
f(X) == AM(AN) |> Mill.data
```

!!! ukn "Model outputs"
    A convenient property of all `Mill.jl` models is that after applying them to a corresponding data node we **always** obtain an `ArrayNode` as output regardless of the type and complexity of the model. This becomes important later.

The most common interpretation of the data inside `ArrayNode`s is that each column contains features of one sample and therefore the node `AN` carries `size(Mill.data(AN), 2)` samples. In this sense, `ArrayNode`s wrap the standard *machine learning* problem, where each sample is represented with a vector, a matrix or a more general tensor of features. Alternatively, one can obtain a number of samples of any `AbstractNode` with `nobs` function from [`StatsBase.jl`](https://github.com/JuliaStats/StatsBase.jl) package:

```@example nodes
using StatsBase: nobs
```

```@repl nodes
nobs(AN)
```

## `BagNode`

`BagNode` is represents the standard *multiple instance learning* problem, that is, each sample is a *bag* containing an arbitrary number of *instances*. In the simplest case, each instance is a vector:

```@repl nodes
BN = BagNode(AN, [1:1, 2:3, 4:4])
```

where for simplicity we used `AN` from the previous example. Each `BagNode` carries `data` and `bags` fields:

```@repl nodes
Mill.data(BN)
BN.bags
```

Here, `data` can be an arbitrary `AbstractNode` storing representation of instances (`ArrayNode` in this case) and `bags` field contains information, which instances belong to which bag. In this specific case `bn` stores three bags (samples). The first one consists of a single instance `{[1.0, 5.0]}` (first column of `AN`), the second one of two instances `{[2.0, 6.0], [3.0, 7.0]}`, and the last one of a single instance `{[4.0, 8.0]}`. We can see that we deal with three top-level samples (bags):

```@repl nodes
nobs(BN)
```

whereas they are formed using four instances:

```@repl nodes
nobs(AN)
```

In `Mill.jl`, there two ways to store indices of the bag's instances:

* in `AlignedBags` structure, which accepts a `Vector` of `UnitRange`s and requires all bag's instances stored continuously:

```@repl nodes
AlignedBags([1:3, 4:4, 5:6])
```

* and in `ScatteredBags` structure, which accepts a `Vector` of `Vectors`s storing not necessarily contiguous indices:

```@repl nodes
ScatteredBags([[3, 2, 1], [4], [6, 5]])
```

The two examples above are semantically equivalent, as bags are unordered collections of instances. An **empty** bag with no instances is in `AlignedBags` specified as empty range `0:-1` and in `ScatteredBags` as an empty vector `Int[]`. The constructor of `BagNode` accepts directly one of these two structures and tries to automagically decide the better type in other cases.

## `BagModel`

Each `BagNode` is processed by a `BagModel`, which contains two (sub)models and an aggregation operator:

```@repl nodes
im = ArrayModel(Dense(2, 3))
a = max_aggregation(3)
bm = ArrayModel(Dense(4, 4))
BM = BagModel(im, a, bm)
```

The first network submodel (called instance model `im`) is responsible for converting the instance representation to a vector form:

```@repl nodes
y = im(AN)
```

Note that because of the property mentioned above, the output of instance model `im` will always be an `ArrayNode` wrapping a matrix. We get four columns, one for each instance. This result is then used in `Aggregation` (`a`) which takes vector representation of all instances and produces a **single** vector per bag:

```@repl nodes
y = a(y, BN.bags)
```

!!! unk "More about aggregation"
    To read more about aggregation operators and find out why there are four rows instead of three after applying the operator, see [Bag aggregation](@ref) section.

Finally, `y` is then passed to a feed forward model (called bag model `bm`) producing the final output per bag. In our example we therefore get a matrix with three columns:

```@repl nodes
y = bm(y)
```

However, the best way to use a bag model node is to simply apply it, which results into the same output:

```@repl nodes
BM(BN) == y
```

The whole procedure is depicted in the following picture:

![](../assets/bagmodel.svg)

Three instances of the `BagNode` are represented by red subtrees are first mapped with instance model `im`, aggregated (aggregation operator here is a concatenation of two different operators ``a_1`` and ``a_2``), and the results of aggregation are transformed with bag model `bm`.

!!! ukn "Musk example"
    Another handy feature of `Mill.jl` models is that they are completely differentiable and therefore fit in the [`Flux.jl`](https://fluxml.ai) framework. Nodes for processing arrays and bags are sufficient to solve the classical [Musk dataset](@ref) problem.

## `ProductNodes` and `ProductModels`

`ProductNode` can be thought of as a [*Cartesian Product*](https://en.wikipedia.org/wiki/Cartesian_product) or a `Dictionary`. It holds a `Tuple` or `NamedTuple` of nodes (not necessarily of the same type). For example, a `ProductNode` with a `BagNode` and `ArrayNode` as children would look like this:

```@repl nodes
PN = ProductNode((a=ArrayNode(Float32.([1 2 3; 4 5 6])), b=BN))
```

Analogically, the `ProductModel` contains a (`Named`)`Tuple` of (sub)models processing each of its children (stored in `ms` field standing for models), as well as one more (sub)model `m`:

```@repl nodes
ms = (a=AM, b=BM)
m = ArrayModel(Dense(7, 2))
PM = ProductModel(ms, m)
```

Again, since the library is based on the property that the output of each model is an `ArrayNode`, the product model applies models from `ms` to appropriate children and vertically concatenates the output, which is then processed by model `m`. An example of model processing the above sample would be:

```@repl nodes
y = PM.m(vcat(PM[:a](PN[:a]), PM[:b](PN[:b])))
```

which is equivalent to:

```@repl nodes
PM(PN) == y
```

Application of a product model (this time with four subtrees (keys)) can be visualized as follows:

![](../assets/productmodel.svg)

!!! unk "Indexing in product nodes"
    In general, we recommend to use `NamedTuple`s, because the key can be used for indexing both `ProductNode`s and `ProductModel`s.
