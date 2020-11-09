**Mill.jl** thinly wraps data and models, such that complicated hierarchies can be build.
The core idea is that data are wrapped to `DataNode <:AbstractNode`. To each `DataNode` corresponds to a `ModelNode <: AbstractModelNode`, such that output of `ModelNode` on `DataNode` is always an `ArrayNode`, which encapsulates a `Matrix` with features. By doing so, we ensure that the models know, what to expect if they rely on other models. 

Below we will go through implementation of `ArrayNode`, `BagNode` and `ProductNode` together with their models. These three types are sufficient to represent any JSON file and using corresponding models to convert it to vector represention (or classify it)

## ArrayNode and ArrayModel

`ArrayNode` wraps a feature matrix. Similarly `ArrayModel` wraps any function performing operation over this feature matrix. In example below, we wrap feature matrix `x` and `Flux.Dense` model.
```julia
using Mill, Flux
ds = ArrayNode(Float32.([1 2 3; 4 5 6]))
m = ArrayModel(Dense(2,3,relu))
m(ds).data == m.m(ds.data)
```

As mentioned, the ArrayNode supports "slicing" and concatenation as follows
```julia
julia> ds[1].data
2×1 Array{Float32,2}:
 1.0
 4.0

julia> ds[[1,3]].data
2×2 Array{Float32,2}:
 1.0  3.0
 4.0  6.0

julia> catobs(ds[1], ds[3]).data
2×2 Array{Float32,2}:
 1.0  3.0
 4.0  6.0
```
which is useful for creating minibatches and their permutations.

## BagNode and BagModel
`BagNode` is the darling of the Mill library, as it implements the basic mill problem. BagModel wraps some node `<:AbstractNode` and information, which instances belongs to which bag which we assume to be independent. Continuing with the above example
```julia
julia> ds = BagNode(ArrayNode(Float32.([1 2 3; 4 5 6])), [1:2, 0:-1, 3:3])
BagNode with 3 bag(s)
  └── ArrayNode(2, 3)
```
defines a `BagNode` which contains three bags. The first one contains two instances `{(1,4), (2,5)}`, the second is empty, and the third one contains single instance `{(3,6)}`. Again, the `BagNode` supports indexing and concatenation

```julia
julia> ds[1]
BagNode with 1 bag(s)
  └── ArrayNode(2, 2)

julia> ds[[1,3]]
BagNode with 2 bag(s)
  └── ArrayNode(2, 3)

julia> catobs(ds[1], ds[3])
BagNode with 2 bag(s)
  └── ArrayNode(2, 3)
```
In this example, we have already shown an important feature, which is handling missing values. More on this topic is in **TODO**.

`BagNode` is processed by `BagModel`, which contains a two neural networks and an aggregation operator. The first network (`im` for instance model) converts the data of bag to a matrix. This matrix representation is used in aggreagation which produces a single vector per bag. This matrix is then passed to another feed forward model (`bm` for bag model) producing the final output.  For the above simple `BagNode`, the corresponding `BagModel` with `mean` aggregation function would look like
```julia
julia> m = BagModel(ArrayModel(Dense(2,3)),
           SegmentedMean(3),
           Dense(3,4)
       )
BagModel ↦ SegmentedMean(3) ↦ ArrayModel(Dense(3, 4))
  └── ArrayModel(Dense(2, 3))
julia> m(ds)
ArrayNode(4, 3)
```
where `im = ArrayModel(Dense(2,3))` and `bm = Dense(3,4)`.

Notice that even though the `BagNode` contained a simple `Matrix`, it has been wrapped in `ArrayNode` and the same holds for `BagModel`. This is important for consistency, since the basic assumption of the library is that model applied on corresponding data produces an `ArrayNode`. That means that the `BagNode` expects that `m.im(ds.data)` returns an `ArrayNode` with a single observation per instances. 

## ProductNodes and ProductModels

`ProductNode` can be thougt as about a Cartesian product or a Dictionary. It holds a `Tuple` or `NamedTuple` of nodes. For example a `ProductNode` with a `BagNode` and `ArrayNode` as childs would look like
```julia
julia> ds = ProductNode(
    (a = BagNode(ArrayNode(Float32.([1 2 3; 4 5 6])), [1:2, 0:-1, 3:3]),
    b = ArrayNode(Float32.([4 5 6; 1 2 3]))))
ProductNode
  ├── a: BagNode with 3 bag(s)
  │        └── ArrayNode(2, 3)
  └── b: ArrayNode(2, 3)
```

Similarly, the `ProductModel` contains a (`Named`)`Tuple` of models procesing in childs (stored in `ms` standing for models). Again, since the library is based on the property that output of model is an `ArrayNode`, the product model applies models from `ms` to appropriate nodes from `ds.data` and vertically concatenates the output, which is then processed by a feedforward network in `m`. An example of model processing the above sample would be
```julia
m = ProductModel((
    a = BagModel(ArrayModel(Dense(2,3)),
           SegmentedMean(3),
           Dense(3,4)
       ),
    b = ArrayModel(Dense(2,3)),
    ),
    ArrayModel(Dense(7,5))
    )

julia> m(ds)
ArrayNode(5, 3)
```

In general, we recommend to use `NamedTuple`, because `key` is used to index models in `ProductModel`.

## Nesting of nodes
Recall the basic design decision, that output of each model on appropriate data is always an `ArrayNode`, the nesting of nodes is a key feature. In the example below, we nest two `BagNode`s and create the appropriate model.
```julia
julia> ds = BagNode(BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),[1:1,2:3,4:5])
BagNode with 3 bag(s)
  └── BagNode with 5 bag(s)
        └── ArrayNode(4, 10)

julia> m = BagModel(
    BagModel(
        ArrayModel(Dense(4, 3, Flux.relu)),   
        SegmentedMeanMax(3),
        ArrayModel(Dense(6, 3, Flux.relu))),
    SegmentedMeanMax(3),
    ArrayModel(Chain(Dense(6, 3, Flux.relu), Dense(3,2))))

BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Chain(Dense(6, 3, relu), Dense(3, 2)))
  └── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        └── ArrayModel(Dense(4, 3, relu))

julia> m(ds)

ArrayNode(2, 3)
```