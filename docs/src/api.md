# API

### Contents

```@contents
Pages = ["api.md"]
```

### Index

```@index
Pages = ["api.md"]
```

## General

```@docs
Mill.emptyismissing
Mill.emptyismissing!

Mill.bagcount
Mill.bagcount!

Mill.string_start_code
Mill.string_start_code!

Mill.string_end_code
Mill.string_end_code!
```

## Bags

```@docs
AbstractBags

AlignedBags
AlignedBags()
AlignedBags(::UnitRange{<:Integer}...)
AlignedBags(::Vector{<:Integer})

ScatteredBags
ScatteredBags()
ScatteredBags(::Vector{<:Integer})

length2bags
bags
remapbags
adjustbags
```

## Data nodes

```@docs
AbstractNode
AbstractProductNode
AbstractBagNode

Mill.data
Mill.metadata
catobs
Mill.subset
Mill.mapdata
removeinstances

ArrayNode
ArrayNode(::AbstractArray)

BagNode
BagNode(::AbstractNode, ::AbstractVector, m)

WeightedBagNode
WeightedBagNode(::AbstractNode, ::AbstractVector, ::Vector, m)

ProductNode
ProductNode(::Any)

LazyNode
LazyNode(::Symbol, ::Any)

Mill.unpack2mill
```

## Special arrays

```@docs
MaybeHotVector
maybehot
MaybeHotMatrix
maybehotbatch

NGramIterator
NGramIterator(::AbstractString, ::Any, ::Any, ::Any)
ngrams
ngrams!
countngrams
countngrams!
NGramMatrix
NGramMatrix(::Missing)

PostImputingMatrix
PostImputingMatrix(::AbstractMatrix)
postimputing_dense

PreImputingMatrix
PreImputingMatrix(::AbstractMatrix)
preimputing_dense
```
