# Data nodes

## Index
```@index
Pages = ["data_nodes.md"]
```

## API
```@docs
AbstractMillNode
AbstractProductNode
AbstractBagNode

ArrayNode
ArrayNode(::AbstractArray)

BagNode
BagNode(::AbstractMillNode, ::AbstractVector, m)

WeightedBagNode
WeightedBagNode(::AbstractMillNode, ::AbstractVector, ::Vector, m)

ProductNode
ProductNode(; ns...)

LazyNode
LazyNode(::Symbol, ::Any)

Mill.unpack2mill

Mill.data
Mill.metadata
datasummary
dropmeta
catobs
Mill.subset
Mill.mapdata
removeinstances

```

