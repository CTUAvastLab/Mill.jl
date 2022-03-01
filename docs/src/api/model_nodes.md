# Model nodes

## Index
```@index
Pages = ["model_nodes.md"]
```

## API
```@docs
AbstractMillModel

ArrayModel

BagModel
BagModel(::AbstractMillModel, ::Union{AbstractAggregation, BagCount}, ::ArrayModel)

ProductModel
ProductModel(::Any)

LazyModel
LazyModel(::Symbol, ::AbstractMillModel)

reflectinmodel
modelsummary
```
