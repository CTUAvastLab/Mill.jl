```@setup mill 
using Mill
```

# Model Reflection

Since constructions of large models can be a tedious and error-prone process, `Mill.jl` provides `reflectinmodel` function that helps to automate it. The simplest definition accepts only one argument, a sample `ds`, and returns a compatible model:

```@repl mill
ds = BagNode(ProductNode((BagNode(ArrayNode(randn(4, 10)),
                                  [1:2, 3:4, 5:5, 6:7, 8:10]),
                          ArrayNode(randn(3, 5)),
                          BagNode(BagNode(ArrayNode(randn(2, 30)),
                                          [i:i+1 for i in 1:2:30]),
                                  [1:3, 4:6, 7:9, 10:12, 13:15]),
                          ArrayNode(randn(2, 5)))),
             [1:1, 2:3, 4:5]);
printtree(ds)

m = reflectinmodel(ds);
printtree(m)

m(ds)
```

The sample `ds` serves here as a *specimen* needed to specify a structure of the problem and calculate dimensions.

## Optional arguments

To have better control over the topology, `reflectinmodel` accepts up to two more optional arguments and four keyword arguments:

* The first optional argument expects a function that returns a layer (or a set of layers) given input dimension `d` (defaults to `d -> Flux.Dense(d, 10)`).
* The second optional argument is a function returning aggregation function for `BagModel` nodes (defaults to `d -> SegmentedMean(d)`).

Compare the following example to the previous one:

```@example mill
using Flux
```

```@repl mill
m = reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMax(d));
printtree(m)

m(ds)
```

## Keyword arguments

The `reflectinmodel` allows even further customization. To index into the sample (or model), we can use `printtree(ds; trav=true)` from [HierarchicalUtils.jl](@ref) that prints the sample together with identifiers of individual nodes:

```@example mill
using HierarchicalUtils
```

```@repl mill
printtree(ds; trav=true)
```

These identifiers can be used to override the default construction functions. Note that the output, i.e. the last feed-forward network of the whole model is always tagged with an empty string `""`, which simplifies putting linear layer with an appropriate output dimension on the end. Dictionaries with these overrides can be passed in as keyword arguments:

* `fsm` overrides constructions of feed-forward models
* `fsa` overrides construction of aggregation functions.

For example to specify just the last feed forward neural network:

```@repl mill
reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax(d);
    fsm = Dict("" => d -> Chain(Dense(d, 20, relu), Dense(20, 12)))) |> printtree
```

Both keyword arguments in action:

```@repl mill
reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax(d);
    fsm = Dict("" => d -> Chain(Dense(d, 20, relu), Dense(20, 12))),
    fsa = Dict("Y" => d -> SegmentedMean(d), "g" => d -> SegmentedPNorm(d))) |> printtree
```
