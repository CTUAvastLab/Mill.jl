## Hierarchical utils
Mill.jl uses [HierarchicalUtils.jl](https://github.com/Sheemon7/HierarchicalUtils.jl) which brings a lot of additional features. For instance, if you want to print a non-truncated version of a model, call:

```julia
julia> printtree(m; trunc=Inf)

BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
  └── ProductModel ↦ ArrayModel(Dense(12, 3, relu))
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        │     └── ArrayModel(Dense(4, 3, relu))
        ├── ArrayModel(Dense(3, 3, relu))
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        │     └── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
        │           └── ArrayModel(Dense(2, 3, relu))
        └── ArrayModel(Dense(2, 3, relu))
```

Callling with `trav=true` enables convenient traversal functionality with string indexing:
```julia
julia>  printtree(m; trunc=Inf, trav=true)

BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu)) [""]
  └── ProductModel ↦ ArrayModel(Dense(12, 3, relu)) ["U"]
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu)) ["Y"]
        │     └── ArrayModel(Dense(4, 3, relu)) ["a"]
        ├── ArrayModel(Dense(3, 3, relu)) ["c"]
        ├── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu)) ["g"]
        │     └── BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu)) ["i"]
        │           └── ArrayModel(Dense(2, 3, relu)) ["j"]
        └── ArrayModel(Dense(2, 3, relu)) ["k"]
```

This way any node in the model tree is swiftly accessible, which may come in handy when inspecting model parameters or simply deleting/replacing/inserting nodes to tree (for instance when constructing adversarial samples). All tree nodes are accessible by indexing with the traversal code:.

```julia
julia> m["Y"]

BagModel ↦ ⟨SegmentedMean(3), SegmentedMax(3)⟩ ↦ ArrayModel(Dense(6, 3, relu))
  └── ArrayModel(Dense(4, 3, relu))
```

The following two approaches give the same result:
```julia
julia> m["Y"] === m.im.ms[1]

true
```

Other functions provided by `HierarchicalUtils.jl`:
```
julia> nnodes(m)

9

julia> nleafs(m)

4

julia> NodeIterator(m) |> collect

9-element Array{AbstractMillModel,1}:
 BagModel
 ProductModel
 BagModel
 ArrayModel
 ArrayModel
 BagModel
 BagModel
 ArrayModel
 ArrayModel

julia> LeafIterator(m) |> collect

4-element Array{ArrayModel{Dense{typeof(relu),Array{Float32,2},Array{Float32,1}}},1}:
 ArrayModel
 ArrayModel
 ArrayModel
 ArrayModel

julia> TypeIterator(m, BagModel) |> collect

4-element Array{BagModel{T,Aggregation{2},ArrayModel{Dense{typeof(relu),Array{Float32,2},Array{Float32,1}}}} where T<:AbstractMillModel,1}:
 BagModel
 BagModel
 BagModel
 BagModel
```

... and many others, see [HierarchicalUtils.jl](https://github.com/Sheemon7/HierarchicalUtils.jl).
