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

Mill.data
Mill.metadata
dropmeta
catobs
Mill.subset
Mill.mapdata
removeinstances

ArrayNode
ArrayNode(::AbstractArray)

BagNode
BagNode(::AbstractMillNode, ::AbstractVector, m)

WeightedBagNode
WeightedBagNode(::AbstractMillNode, ::AbstractVector, ::Vector, m)

ProductNode
ProductNode(::Any)

LazyNode
LazyNode(::Symbol, ::Any)

Mill.unpack2mill
```

