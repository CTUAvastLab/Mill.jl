```@setup reflection 
using Mill, Flux
```

# Model reflection

Since constructions of large models can be a tedious and error-prone process, [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) provides [`reflectinmodel`](@ref) function that helps to automate it. The simplest definition accepts only one argument, a sample `ds`, and returns a compatible model:

```@repl reflection
ds = BagNode(ProductNode((BagNode(randn(Float32, 4, 10),
                                  [1:2, 3:4, 5:5, 6:7, 8:10]),
                          randn(Float32, 3, 5),
                          BagNode(BagNode(randn(Float32, 2, 30),
                                          [i:i+1 for i in 1:2:30]),
                                  [1:3, 4:6, 7:9, 10:12, 13:15]),
                          randn(Float32, 2, 5))),
             [1:1, 2:3, 4:5]);
printtree(ds)

m = reflectinmodel(ds, d -> Dense(d, 2));
printtree(m)

m(ds)
```

The sample `ds` serves here as a *specimen* needed to specify a structure of the problem and calculate dimensions.

### Optional arguments

To have better control over the topology, [`reflectinmodel`](@ref) accepts up to two more optional arguments and four keyword arguments:

* The first optional argument expects a function that returns a layer (or a set of layers) given input dimension `d` (defaults to `d -> Flux.Dense(d, 10)`).
* The second optional argument is a function returning aggregation function for [`BagModel`](@ref) nodes (defaults to `BagCount ∘ SegmentedMeanMax`).

Compare the following example to the previous one:

```@repl reflection
m = reflectinmodel(ds, d -> Dense(d, 5, relu), SegmentedMax);
printtree(m)

m(ds)
```

### Keyword arguments

The [`reflectinmodel`](@ref) allows even further customization. To index into the sample (or model),
we can use `printtree(ds; trav=true)` from [`HierarchicalUtils.jl`](@ref) that prints the sample
together with identifiers of individual nodes:

```@example reflection
using HierarchicalUtils
```

```@repl reflection
printtree(ds; trav=true)
```

These identifiers can be used to override the default construction functions. Note that the output, i.e. the last feed-forward network of the whole model is always tagged with an empty string `""`, which simplifies putting linear layer with an appropriate output dimension on the end. Dictionaries with these overrides can be passed in as keyword arguments:

* `fsm` overrides constructions of feed-forward models
* `fsa` overrides construction of aggregation functions.

For example to specify just the last feed forward neural network:

```@repl reflection
reflectinmodel(ds, d -> Dense(d, 5, relu), SegmentedMeanMax;
    fsm = Dict("" => d -> Chain(Dense(d, 20, relu), Dense(20, 12)))) |> printtree
```

Both keyword arguments in action:

```@repl reflection
reflectinmodel(ds, d -> Dense(d, 5, relu), SegmentedMeanMax;
    fsm = Dict("" => d -> Chain(Dense(d, 20, relu), Dense(20, 12))),
    fsa = Dict("Y" => SegmentedMean, "g" => SegmentedPNorm)) |> printtree
```

There are even more ways to modify the reflection behavior, see the [`reflectinmodel`](@ref) api reference.

### `Float` precision

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) is built on top of [`Flux.jl`](https://fluxml.ai),
which by default uses 32-bit precision for model parameters:

```@repl reflection
Dense(2, 2).weight |> eltype
```

If you attempt to process `Float64` data with model using lower precision, you get a warning:

```@repl reflection
x = randn(2, 2)
eltype(x)
m = Dense(2, 2)
m(x)
```

Unless additional arguments are provided, [`reflectinmodel`](@ref) also instantiates all `Dense`
layers using 32-bit precision:

```@repl reflection
x = randn(Float32, 2, 2) |> ArrayNode
eltype(Mill.data(x))
m = reflectinmodel(x)
m.m.weight |> eltype
```

Because [`reflectinmodel`](@ref) evaluates (sub)models on parts of the input when building the
model, if some `Float64` values are passed in, the warning is shown during construction as well as
during the evaluation:

```@repl reflection
x = randn(2, 2) |> ArrayNode
eltype(Mill.data(x))
m = reflectinmodel(x)
m(x)
```

To prevent this from happening, we recommend making sure that the same precision is used for input
data and for [`reflectinmodel`](@ref) parameters. For example:

```@repl reflection
x32 = randn(Float32, 2, 2) |> ArrayNode
m = reflectinmodel(x32)

x64 = randn(2, 2) |> ArrayNode
m = reflectinmodel(x64, d -> f64(Dense(d, 5)), d -> f64(SegmentedMean(d)))
```

Functions `Flux.f64` and `Flux.f32` may come in handy.

