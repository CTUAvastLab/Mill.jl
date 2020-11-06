## More complicated models
The main advantage of the Mill library is that it allows to arbitrarily nest and cross-product `BagModels`, as is described in Theorem 5 of [[3](#cit3)].
 Let's start the demonstration by nesting two MIL problems. The outer MIL model contains three samples. The first sample contains another bag (inner MIL) problem with two instances, the second sample contains two inner bags with total of three instances, and finally the third sample contains two inner bags with four instances.
```julia
julia> ds = BagNode(BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),[1:1,2:3,4:5])
BagNode with 3 bag(s)
  └── BagNode with 5 bag(s)
        └── ArrayNode(4, 10)
```
 We can create the model manually as in the case of Musk as
```julia
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
```
and we can apply the model as
```julia
julia> m(ds)

ArrayNode(2, 3)
```
 Since constructions of large models can be a process prone to errors, there is a function `reflectinmodel` which tries to automatize it keeping track of dimensions. It accepts as a first parameter a sample `ds`. Using the function on the above example creates a model:
```julia
julia> m = reflectinmodel(ds)

BagModel ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10))
  └── BagModel ↦ SegmentedMean(10) ↦ ArrayModel(Dense(10, 10))
        └── ArrayModel(Dense(4, 10))
```

To have better control over the topology, `reflectinmodel` accepts up to four additional parameters. The second parameter is a function returning layer (or set of layers) with input dimension `d`, and the third function is a function returning aggregation functions for `BagModel`:
```julia
julia> m = reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax(d))

BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
  └── BagModel ↦ ⟨SegmentedMean(5), SegmentedMax(5)⟩ ↦ ArrayModel(Dense(10, 5, relu))
        └── ArrayModel(Dense(4, 5, relu))
```

Let's test the model
```julia
julia> m(ds).data

5×3 Array{Float32,2}:
 0.0542484   0.733629  0.553823
 0.062246    0.866254  1.03062 
 0.027454    1.04703   1.63135 
 0.00796955  0.36415   1.18108 
 0.034735    0.17383   0.0
```

 ### Even more complicated models
As already mentioned above, the datasets can contain Cartesian products of MIL and normal (non-MIL) problems. Let's do a quick demo.
```julia
julia> ds = BagNode(
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
```
For this, we really want to create model automatically despite it being sub-optimal.
```julia
julia> m = reflectinmodel(ds, d -> Dense(d, 3, relu), d -> SegmentedMeanMax(d))

BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
  └── ProductModel ↦ ArrayModel(Dense(12, 3, relu))
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        │     ⋮
        ├── ArrayModel(Dense(3, 3, relu))
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        │     ⋮
        └── ArrayModel(Dense(2, 3, relu))
```
