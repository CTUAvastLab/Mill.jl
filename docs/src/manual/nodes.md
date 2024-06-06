```@setup nodes
using Mill
```

!!! ukn "Tip"
    It is recommended to read the [Motivation](@ref) section first to understand the crucial ideas behind hierarchical multiple instance learning.

# Nodes

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) enables representation of arbitrarily complex tree-like hierarchies and appropriate models for these hierarchies. It defines two core abstract types:

1. [`AbstractMillNode`](@ref) which stores data on any level of abstraction and its subtypes can be further nested
2. [`AbstractMillModel`](@ref) which helps to define a corresponding model. For each specific implementation of [`AbstractMillNode`](@ref) we have one or more specific [`AbstractMillModel`](@ref)s for processing it.

Below we will introduce [`ArrayNode`](@ref), [`BagNode`](@ref) and [`ProductNode`](@ref) together with their corresponding models. It is possible to define data and model nodes for more complex behaviors (see [Custom nodes](@ref)), however, these three core types are already sufficient for most tasks. For instance, we can represent any `JSON` document and use appropriate models to convert it to a vector represention or classify it (see [Processing JSONs](@ref)).

## [`ArrayNode`](@ref) and [`ArrayModel`](@ref)

[`ArrayNode`](@ref) thinly wraps an array of features (specifically any subtype of `AbstractArray`):

```@repl nodes
X = Float32.([1 2 3 ; 4 5 6])
AN = ArrayNode(X)
```

Data carried by any [`AbstractMillNode`](@ref) can be accessed with the [`Mill.data`](@ref) function as follows:

```@repl nodes
Mill.data(AN)
```

Similarly, `ArrayModel` wraps any function performing operation over this array. In example below, we wrap a feature matrix `X` and a `Dense` model from [`Flux.jl`](https://fluxml.ai):

```@example nodes
using Flux: Dense
```

```@repl nodes
f = Dense(2, 4)
AM = ArrayModel(f)
```

We can apply the model now with `AM(AN)` to get another [`ArrayNode`](@ref) and verify that the feedforward layer `f` is really applied:

```@repl nodes
AM(AN)
f(X) == AM(AN)
```

!!! ukn "Model outputs"
    A convenient property of all [`AbstractMillModel`](@ref) nodes is that after applying them to a corresponding data node we **always** obtain an array as output regardless of the type and complexity of the model. This becomes important later.

The most common interpretation of the data inside [`ArrayNode`](@ref)s is that each column contains
features of one sample and therefore the node `AN` carries `size(Mill.data(AN), 2)` samples. In this
sense, [`ArrayNode`](@ref)s wrap the standard *machine learning* problem, where each sample is
represented with a vector, a matrix or a more general tensor of features. Alternatively, one can
obtain a number of samples of any [`AbstractMillNode`](@ref) with `numobs` function from
[`MLUtils.jl`](https://github.com/JuliaML/MLUtils.jl) package, which
[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) also exports:

```@repl nodes
numobs(AN)
```

## [`BagNode`](@ref)

[`BagNode`](@ref) represents the standard *multiple instance learning* problem, that is, each sample is a *bag* containing an arbitrary number of *instances*. In the simplest case, each instance is a vector:

```@repl nodes
BN = BagNode(AN, [1:2, 0:-1, 3:3])
```

where for simplicity we used `AN` from the previous example. It is also possible to use data
directly, in such case it is wrapped in an [`ArrayNode`](@ref) automatically.

Each [`BagNode`](@ref) carries `data` and `bags` fields:

```@repl nodes
Mill.data(BN)
BN.bags
```

Here, `data` can be an arbitrary [`AbstractMillNode`](@ref) storing representation of instances ([`ArrayNode`](@ref) in this case) and `bags` field contains information, which instances belong to which bag. In this specific case `bn` stores three bags (samples). The first one consists of a two instances `{[1.0, 4.0], [2.0, 5.0]}` (first two columns of `AN`), the second one is empty, and the thirs bag contains one instance `{[3.0, 6.0]}`. We can see that we deal with two top-level samples (bags):

```@repl nodes
numobs(BN)
```

whereas they are formed using three instances:

```@repl nodes
numobs(AN)
```

In [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl), there are two ways to store indices of the
bag's instances:

* in [`AlignedBags`](@ref) structure, which accepts a `Vector` of `UnitRange`s and requires all bag's instances stored continuously:

```@repl nodes
AlignedBags([1:2, 3:3])
```

* and in [`ScatteredBags`](@ref) structure, which accepts a `Vector` of `Vectors`s storing not necessarily contiguous indices:

```@repl nodes
ScatteredBags([[2, 1], [3]])
```

The two examples above are semantically equivalent, as bags are unordered collections of instances. An **empty** bag with no instances is in [`AlignedBags`](@ref) specified as empty range `0:-1` and in [`ScatteredBags`](@ref) as an empty vector `Int[]`. The constructor of [`BagNode`](@ref) accepts directly one of these two structures and tries to automagically decide the better type in other cases.

## [`BagModel`](@ref)

Each [`BagNode`](@ref) is processed by a [`BagModel`](@ref), which contains two (sub)models and an aggregation operator:

```@repl nodes
im = ArrayModel(Dense(2, 5))
a = SegmentedMax(5)
bm = Dense(5, 3)
BM = BagModel(im, a, bm)
```

The first network submodel (called instance model `im`) is responsible for converting the instance representation to a vector form:

```@repl nodes
y = im(AN)
```

Note that because of the property mentioned above, the output of instance model `im` will always be a `Matrix`. We get four columns, one for each instance. This result is then used in [`SegmentedMax`](@ref) operator `a` which takes vector representation of all instances and produces a **single** vector per bag:

```@repl nodes
y = a(y, BN.bags)
```

!!! unk "More about aggregation"
    To read more about aggregation operators, see the [Bag aggregation](@ref) section. For an
    explanation how empty bags are aggregated, see [Missing data](@ref).

Finally, `y` is then passed to a feed forward model (called bag model `bm`) producing the final output per bag. In our example we therefore get a matrix with three columns:

```@repl nodes
y = bm(y)
```

However, the best way to use a bag model node is to simply apply it, which results into the same output:

```@repl nodes
BM(BN) == y
```

The whole procedure is depicted in the following picture:

```@raw html
<img class="display-light-only" src="../../assets/bagmodel.svg" alt="Bag Model"/>
<img class="display-dark-only" src="../../assets/bagmodel-dark.svg" alt="Bag Model"/>
```

Three instances of the [`BagNode`](@ref) are represented by red subtrees are first mapped with instance model `im`, aggregated (aggregation operator here is a concatenation of two different operators ``a_1`` and ``a_2``), and the results of aggregation are transformed with bag model `bm`.

!!! ukn "Musk example"
    Another handy feature of [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) models is that they
    are completely differentiable and therefore fit in the [`Flux.jl`](https://fluxml.ai) framework.
    Nodes for processing arrays and bags are sufficient to solve the classical [Musk](@ref) problem.

## [`ProductNode`](@ref)s and [`ProductModel`](@ref)s

[`ProductNode`](@ref) can be thought of as a [Cartesian Product](https://en.wikipedia.org/wiki/Cartesian_product) or a `Dictionary`. It holds a `Tuple` or `NamedTuple` of nodes (not necessarily of the same type). For example, a [`ProductNode`](@ref) with the [`BagNode`](@ref) and the [`ArrayNode`](@ref) from above and two more nodes as children would look like this:

```@repl nodes
PN = tuple(
        ArrayNode(randn(Float32, 2, 3)),
        BagNode(ArrayNode(zeros(Float32, 4, 4)), [1:1, 2:2, 3:4]),
        BN,
        AN
    ) |> ProductNode
```

Analogically, the [`ProductModel`](@ref) contains a `Tuple` or `NamedTuple` of (sub)models processing each of its children (stored in `ms` field standing for models), as well as one more (sub)model `m`:

```@repl nodes
ms = tuple(
        ArrayModel(Dense(2, 2)),
        BagModel(ArrayModel(Dense(4, 6)), SegmentedMean(6), Dense(6, 5)),
        BM,
        AM);
m = Dense(14, 9);
PM = ProductModel(ms, m)
```

Again, since the library is based on the property that the output of each model is an array, the product model applies models from `ms` to appropriate children and vertically concatenates the output, which is then processed by model `m`. An example of model processing the above sample would be:

```@repl nodes
y = PM.m(vcat([PM.ms[i](PN.data[i]) for i in 1:4]...))
```

which is equivalent to:

```@repl nodes
PM(PN) == y
```

Application of this product model can be schematically visualized as follows:

```@raw html
<img class="display-light-only" src="../../assets/productmodel.svg" alt="Product Model"/>
<img class="display-dark-only" src="../../assets/productmodel-dark.svg" alt="Product Model"/>
```
