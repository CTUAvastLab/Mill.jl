```@setup musk
using Random; Random.seed!(42)

using Pkg
old_path = Pkg.project().path
Pkg.activate(pwd())
Pkg.instantiate()
```

# Musk dataset
[`Musk dataset`](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)) is a classic MIL problem of the field, introduced in [Dietterich1997](@cite). Below we demonstrate how to solve this problem using `Mill.jl`. The full example is also accessible [here](https://github.com/pevnak/Mill.jl/tree/master/docs/src/examples/musk/), as well as a Julia environment to run it.

For the demo, we load all dependencies:

```@example musk
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
```

and then load the dataset and transform it into a `Mill.jl` structure. The `musk.jld2` file contains:

* a matrix with features `fMat`:

```@repl musk
fMat = load("musk.jld2", "fMat")          # matrix with instances, each column is one sample
```

* the id of sample (*bag* in MIL terminology) specifying to which each instance (column in `fMat`) belongs to:

```@repl musk
bagids = load("musk.jld2", "bagids")      # ties instances to bags
```

The resulting `BagNode` is a structure which holds (i) feature matrix and (ii) ranges identifying which columns in the feature matrix each bag spans. This representation ensures that feed-forward networks do not need to deal with bag boundaries and always process full continuous matrices:

```@repl musk
ds = BagNode(ArrayNode(fMat), bagids)     # create a BagNode dataset
```

* the label of each instance in `y`.  The label of a bag is a maximum of labels of its instances, i.e. one positive instance in a bag makes it positive:

```@repl musk
y = load("musk.jld2", "y")                # load labels
y = map(i -> maximum(y[i]) + 1, ds.bags)  # create labels on bags
y_oh = Flux.onehotbatch(y, 1:2)           # one-hot encoding
```

Once the data are in `Mill.jl` internal format, we will manually create a model. `BagModel` is designed to implement a basic multi-instance learning model utilizing two feed-forward networks with an aggregaton operator in between:

```@repl musk
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.relu)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(21, 10, Flux.relu), Dense(10, 2))))  # model on the level of bags
```

Instances are first passed through a single layer with 10 neurons (input dimension is 166) with `relu` non-linearity, then we use `mean` and `max` aggregation functions simultaneously (for some problems, max is better then mean, therefore we use both), and then we use one layer with 10 neurons and `relu` nonlinearity followed by linear layer with 2 neurons (output dimension).

Let's check that forward pass works:

```@repl musk
model(ds)
```

Since `Mill.jl` is entirely compatible with [`Flux.jl`](https://fluxml.ai), we can use its `cross-entropy` loss function:

```@repl musk
loss(ds, y_oh) = Flux.logitcrossentropy(model(ds).data, y_oh)
```

and run simple training procedure using its tooling:

```@repl musk
opt = Flux.ADAM()
@epochs 10 begin
    Flux.train!(loss, params(model), repeated((ds, y_oh), 1000), opt)
    println(loss(ds, y_oh))
end
```

We can also calculate training error, which should be not so surprisingly low:

```@repl musk
mean(mapslices(argmax, model(ds).data, dims=1)' .!= y)
```

```@setup musk
Pkg.activate(old_path)
```
