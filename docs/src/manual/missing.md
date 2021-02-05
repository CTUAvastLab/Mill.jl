```@setup missing
using Mill
```

# Missing data

One detail that was left out so far is how `Mill.jl` handles incomplete or missing data. This phenomenon is nowadays ubiquitous in many data sources and occurs due to:

* a high price of obtaining a (part of) observation
* information being unreachable due to privacy reasons
* a gradual change in the definition of data being gathered
* a faulty collection process

and many other possible reasons. At the same time, it is wasteful to throw away the incomplete observations altogether. Thanks to the hierarchical structure of both samples and models, we can still process samples with missing information fragments at various levels of abstraction. Problems of this type can be categorized into 3 not necessarily separate types:

1. Missing parts of raw-data in a leaf `ArrayNode`
2. Empty bags with no instances in a `BagNode`
3. And entire key missing in a `ProductNode`

At the moment, `Mill.jl` is capable of handling the first two cases. The solution always involves an additional vector of parameters (denoted always by `ψ`) that are used during the model evaluation to substitute the missing values. Parameters `ψ` can be either fixed or learned during training. Everything is done automatically.

## Empty bags

It may happen that some bags in the datasets are empty by definition or no associated instances were obtained during data collection. Recall, that an empty bag is specified as empty range `0:-1` in case of `AlignedBags` and as an empty vector `[]` when `ScatteredBags` are used:

```@repl missing
empty_bags_1 = AlignedBags([1:2, 0:-1, 3:5, 0:-1])
empty_bags_2 = ScatteredBags([[1, 2], [], [3, 4, 5], []])
```

To obtain the vector representation for a bag, be it for dircetly predicting some value or using it to represent some higher-level structures, we need to deal with these empty bags. This is done in [`Bag aggregation`](@ref). Each `AggregationOperator` carries a vector of parameters `ψ`, initialized to zeros upon creation:

```@repl missing
a = SegmentedSumMax(2)
```

When we evaluate any `BagModel`, these values are used to compute output for empty bags instead of the aggregation itself. See the demo below:

```@repl missing
an = ArrayNode(randn(Float32, 2, 5))
ds = BagNode(an, empty_bags_2)
m = BagModel(identity, a, identity)
m(ds)
```

Vector `ψ` is learnable and therefore after training will contain a suitable representation of an empty bag for the given problem.

When a `BagNode` is entirely empty, it can be constructed with `missing` instead of a matrix wrapped in an `ArrayNode`:

```@repl missing
bn1 = BagNode(ArrayNode(rand(3, 4)), [1:4])
bn2 = BagNode(missing, [0:-1])
```

and everything will work as expected. For example, we can concatenate these two:

```@repl missing
x = catobs(bn1, bn2)
```

Notice, that the resulting `ArrayNode` has still the same dimension as `ArrayNode` inside `bn1`. The emptiness of `bn2` is stored in `bags`:

```@repl missing
x.bags
```

The second element `BagNode` can be obtained again by indexing:

```@repl missing
bn1 == x[2]
```

Even though this approach of using `missing` for `data` field in `BagNodes` is the most accurate from the semantic point of view, it may cause excessive compilation, as the types will be different. Therefore, if this happens in multiple places in the sample tree, it may be better to instead use an empty matrix for type consistency:

```@repl missing
BagNode(ArrayNode(zeros(3, 0)), [0:-1])
```

How indexing behaves with respect to this issue depends on a global switch (off by default) and 
can be changed with the [`Mill.emptyismissing!`](@ref) function:

```@repl missing
a = BagNode(ArrayNode(rand(3, 2)), [1:2, 0:-1, 0:-1])
a[2:3] |> Mill.data
Mill.emptyismissing!(true)
a[2:3] |> Mill.data
missing
```

```@setup missing
Mill.emptyismissing!(false)
```

## `PostImputingMatrix`

Storing missing strings in `NGramMatrix` is straightforward:

```@repl missing
missing_ngrams = NGramMatrix(["foo", missing, "bar"], 3, 256, 5)
```

When some values of categorical variables are missing, `Mill.jl` defines a new type for representation:

```@repl missing
missing_categorical = maybehotbatch([missing, 2, missing], 1:5)
```

`MaybeHotMatrix` behaves similarly as `OneHotMatrix` from [`Flux.jl`](https://fluxml.ai), but it supports possibly `missing` values. In case when no values are `missing` it behaves exactly like `OneHotMatrix`:

```@repl missing
maybehotbatch([5, 2, 1], 1:5)
```

`MaybeHotMatrix` behaves like `AbstractMatrix` and supports left multiplication again:

```@repl missing
missing_categorical::AbstractMatrix{Union{Bool, Missing}}
```

However, multiplying these matrices with `missing` data leads into `missing` data in the output.

```@repl missing
W = rand(2, 5)
W * missing_ngrams
W * missing_categorical
```

Consequently, gradient can't be computed and any model can't be trained.

!!! ukn "Model debugging"
    Flux `gradient` call returns an error like `Output should be scalar; gradients are not defined for output missing` when attempted on `missing` result. In a similar fashion as having `NaN`s in a model, this signifies that some `missing` input is not treated anywhere in the model and it propagates up. Generally speaking, it is recommended to deal with missing values as soon as possible (on the leaf level) so that they do not propagate and cause type instabilities.

`PostImputingMatrix` is a solution for this. It can be constructed as follows:

```@repl missing
A = PostImputingMatrix(W)
```

Matrix `W` is stored inside and `A` creates one vector of parameters `ψ` of length `size(W, 1)` on top of that. Suddenly, multiplication automagically works:

```@repl missing
A * missing_ngrams
A * missing_categorical
```

What happens under the hood is that whenever `A` encounters a `missing` column in the matrix, it fills in values from `ψ` **after** the multiplication is performed (effectively replacing all `missing` values in the result of multiplying with `W`, but implemented more efficiently). Vector `ψ` can be learned during training as well and everything works out of the box.

## `PreImputingMatrix`

If we have to deal with inputs where some elements of input matrix are `missing`:

```@repl missing
X = [missing 1 2; 3 missing missing]
```

we can make use of `PreImputingMatrix`:

```@repl missing
W = rand(1:2, 3, 2)
A = PreImputingMatrix(W)
```

As opposed to `PostImputingMatrix`, `A` now stores a vector of values `ψ` with length `size(W, 2)`. When we use it for multiplication:

```@repl missing
A * X
```

what happens is that when we perform a dot product of a row of `A` and a column of `X`, we first fill in values from `ψ` into the column **before** the multiplication is performed. Again, it is possible to compute gradients with respect to all three of `W`, `ψ` and `X` and therefore learn the appropriate default values in `ψ` from the data:

```@repl missing
using Flux

gradient((A, X) -> sum(A * X), A, X)
```

## Model reflection with missing values

Model reflection takes `missing` values and types into account and creates appropriate (sub)models to handle them:

```@repl missing
ds = ProductNode(ArrayNode.((missing_ngrams, missing_categorical, X)))
m = reflectinmodel(ds)
```

Here, `[pre_imputing]Dense` and `[post_imputing]Dense` are standard dense layers with a special matrix inside:

```@repl missing
dense = m.ms[1].m; typeof(dense.W)
```

Inside `Mill.jl` we add a special definition `Base.show` for these types for compact printing.

The `reflectinmodel` method use types to determine whether imputing is needed or not. Compare the following:

```@repl missing
reflectinmodel(ArrayNode(randn(2, 3)))
reflectinmodel(ArrayNode([1.0 2.0 missing; 4.0 missing missing]))
reflectinmodel(ArrayNode(Matrix{Union{Missing, Float64}}(randn(2, 3))))
```

In the last case, the imputing type is returned even though there is no `missing` element in the matrix. Of course, the same applies to `MaybeHot*` types and `NGramMatrix`. This way, we can signify that even though there are no missing values in the available sample, we expect them to appear in the future and want our model compatible. If it is hard to determine this in advance a safe bet is to make all leaves in the model. The performance will not suffer because imputing types are as fast as their non-imputing counterparts on data not containing `missing` values and the only tradeoff is a slight increase in the number of parameters, some of which may never be used.
