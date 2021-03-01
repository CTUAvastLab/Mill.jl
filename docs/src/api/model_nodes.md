# Model nodes

## Index
```@index
Pages = ["model_nodes.md"]
```

## API
```@docs
AbstractMillModel

ArrayModel
identity_model
IdentityModel

BagModel
BagModel(::AbstractMillModel, ::Aggregation, ::ArrayModel)

ProductModel
ProductModel(::Tuple{AbstractMillModel}, ::ArrayModel)

LazyModel
LazyModel(::Symbol, ::AbstractMillModel)

reflectinmodel
```
