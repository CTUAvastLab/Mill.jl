# Missing data

With the latest version of Mill, it is also possible to work with missing data, replacing a missing bag with a default constant value, and even to learn this value as well. Everything is done automatically.

At the moment, Mill.jl features an initial and naive approach to missing values. We assume that `ArrayNode` have missing values replaced by zeros, which is not optimal but in many situations it works reasonably well.

BagNodes with missing features are indicated by Bags being set to `[0:-1]` with `missing` as a data and metadata. This can be seamlessly concatenated or sub-set, if the operation makes sense.

Couple examples from unit tests. Let's define full and empty BagNode
```
julia> a = BagNode(ArrayNode(rand(3,4)),[1:4], nothing)
BagNode with 1 bag(s)
  └── ArrayNode(3, 4)

julia> e = BagNode(missing, AlignedBags([0:-1]), nothing)
BagNode with 1 empty bag(s)
```

We can concatenate them as follows.
```
julia> x = reduce(catobs,[a, e])
BagNode with 2 bag(s)
  └── ArrayNode(3, 4)
```
Notice, that the ArrayNode has still the same dimension as ArrayNode of just `a`. The missing second element, corresponding to `e` is indicated by the second bags being `0:-1` as follows:
```
julia> x.bags
AlignedBags(UnitRange{Int64}[1:4, 0:-1])
```

We can get back the missing second element as
```
julia> x[2]
BagNode with 1 empty bag(s)
```

During forward (and backward) pass, the missing values in BagNodes are filled in aggregation by zeros. ** In order this feature to work, the `Aggregation` needs to know dimension, therefore use MissingAggregation, which can handle this.** In the future, MissingAggregation will be made default.

Last but not least,
`ProductNode`s cannot handle missing values, as the missingness is propagated to its leaves, i.e.
```
julia> ProductNode((a,e))
ProductNode{2}
  ├── BagNode with 1 bag(s)
  │     └── ArrayNode(3, 4)
  └── BagNode with 1 empty bag(s)
```

## Representing missing values
The library currently support two ways to represent bags with missing values. First one represent missing data using `missing` as `a = BagNode(missing, [0:-1])` while the second as an empty vector as `a = BagNode(zero(4,0), [0:-1])`.  While off the shelf the library supports both approaches transparently, the difference is mainly when one uses getindex, and therefore there is a switch
`Mill.emptyismissing(false)`, which is by default false. Let me demonstrate the difference.
```julia 
julia> a = BagNode(ArrayNode(rand(3,2)), [1:2, 0:-1])
BagNode with 2 bag(s)
  └── ArrayNode(3, 2)

julia> Mill.emptyismissing(false);

julia> a[2].data
ArrayNode(3, 0)

julia> Mill.emptyismissing(true)
true

julia> a[2].data
missing
```
The advantage of the first approach, default, is that types are always the same, which is nice to the compiler (and Zygote). The advantage of the latter is that it is more compact and nicer.
