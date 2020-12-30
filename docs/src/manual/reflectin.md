## ReflectInModel
Constructions of large models can be boring and a process prone to errors. Therefore `Mill.jl` comes with a convenient function `reflectinmodel` which does this for you. The result is certainly suboptimal, which can be said about almost any Neural Network. The reflect in model accepts three important parameters.
1. The data sample serving as a specimen, which is needed to calculate dimensions and know the structure;
2. a function returning a feed-forward model (or a function) accepting data in form of a Matrix with dimension `d`;
3. a function returing aggregation function in `BagModel`s.

For example to create a model for the below sample
```julia
ds = BagNode(
    ProductNode(
        (BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),
        ArrayNode(randn(3,5)),
        BagNode(
            BagNode(ArrayNode(randn(2,30)),[i:i+1 for i in 1:2:30]),
            [1:3,4:6,7:9,10:12,13:15]),
        ArrayNode(randn(2,5)))),
    [1:1,2:3,4:5])

BagNode with 3 bag(s)
  └── ProductNode
        ├── BagNode with 5 bag(s)
        │     ⋮
        ├── ArrayNode(3, 5)
        ├── BagNode with 5 bag(s)
        │     ⋮
        └── ArrayNode(2, 5)

julia> m = reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax(d))

BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
  └── ProductModel ↦ ArrayModel(Dense(20, 5, relu))
        ├── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        │     ⋮
        ├── ArrayModel(Dense(3, 5, relu))
        ├── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        │     ⋮
        └── ArrayModel(Dense(2, 5, relu))

julia> m(ds)
ArrayNode(5, 3)
```

The `reflectinmodel` allows customization. To index into the sample, `printtree(ds, trav = true)` prints the sample with identifiers identifying invidual nodes. For example
```julia
julia> Mill.printtree(ds, trav = true)
BagNode with 3 bag(s) [""]
  └── ProductNode ["U"]
        ├── BagNode with 5 bag(s) ["Y"]
        │     └── ArrayNode(4, 10) ["a"]
        ├── ArrayNode(3, 5) ["c"]
        ├── BagNode with 5 bag(s) ["g"]
        │     └── BagNode with 15 bag(s) ["i"]
        │           └── ArrayNode(2, 30) ["j"]
        └── ArrayNode(2, 5) ["k"]
```
These identifiers can be used to override the default construction functions. Note that the output, i.e. the last feed-forward network of the whole model is always tagged `""`, which simpliefies putting linear layer with appropriate out dimension on the end. These overrides are put to dictionaries with `b` overriding constructions of feed-forward models and `a` overriding construction of aggregation functions. For example to specify just last Feed Forward Neural Network
```julia
julia> m = reflectinmodel(ds, 
    d -> Dense(d, 5, relu), 
    d -> SegmentedMeanMax(d),
    b = Dict("" => d -> Chain(Dense(d, 20,relu), Dense(20,12))),
    )

BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Chain(Dense(10, 20, relu), Dense(20, 12)))
  └── ProductModel ↦ ArrayModel(Dense(20, 5, relu))
        ├── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        │     ⋮
        ├── ArrayModel(Dense(3, 5, relu))
        ├── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        │     ⋮
        └── ArrayModel(Dense(2, 5, relu))
```

To change the aggregation function in the middle (for some obscure reason)
```julia
m = reflectinmodel(ds, 
    d -> Dense(d, 5, relu), 
    d -> SegmentedMeanMax(d),
    b = Dict("" => d -> Chain(Dense(d, 20,relu), Dense(20,12))),
    a = Dict("Y" => d -> SegmentedMean(d), "g" => d -> SegmentedMean(d))
    );
julia> printtree(m)
BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Chain(Dense(10, 20, relu), Dense(20, 12)))
  └── ProductModel ↦ ArrayModel(Dense(20, 5, relu))
        ├── BagModel ↦ SegmentedMean(5) ↦ ArrayModel(Dense(5, 5, relu))
        │     └── ArrayModel(Dense(4, 5, relu))
        ├── ArrayModel(Dense(3, 5, relu))
        ├── BagModel ↦ SegmentedMean(5) ↦ ArrayModel(Dense(5, 5, relu))
        │     └── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        │           └── ArrayModel(Dense(2, 5, relu))
        └── ArrayModel(Dense(2, 5, relu))
```
