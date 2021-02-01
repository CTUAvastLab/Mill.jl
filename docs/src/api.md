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
```
