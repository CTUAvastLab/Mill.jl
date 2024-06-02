```@setup musk
using Pkg
old_path = Pkg.project().path
Pkg.activate(pwd())
Pkg.instantiate()
```
```@meta
EditURL = "musk_literate.jl"
```

# Musk

[`Musk dataset`](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)) is a classic MIL problem of the field, introduced in [Dietterich1997](@cite). Below we demonstrate how to solve this problem using [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl).
!!! ukn "Jupyter notebook"
    This example is also available as a [Jupyter notebook](<unknown>/examples/musk/musk.ipynb)
    and the environment is accessible [here](https://github.com/CTUAvastLab/Mill.jl/tree/master/docs/src/examples/musk).

We load all dependencies and fix the seed:

````@example musk
using FileIO, JLD2, Statistics, Mill, Flux, OneHotArrays

using Random; Random.seed!(42);
nothing #hide
````

### Loading the data

Now we load the dataset and transform it into a `Mill` structure. The `musk.jld2` file contains...
* a matrix with features, each column is one instance:

````@example musk
fMat = load("musk.jld2", "fMat")
````

* the ids of samples (*bags* in MIL terminology) specifying to which each instance (column in `fMat`) belongs to:

````@example musk
bagids = load("musk.jld2", "bagids")
````

* and labels defined on the level of instances:

````@example musk
y = load("musk.jld2", "y")
````

We create a [`BagNode`](@ref) structure which holds:
1. feature matrix and
2. ranges identifying which columns in the feature matrix each bag spans.

````@example musk
ds = BagNode(ArrayNode(fMat), bagids)
````

This representation ensures that feed-forward networks do not need to deal with bag boundaries and always process full continuous matrices:

We also compute labels on the level of bags. In the `Musk` problem, bag label is defined as a maximum of instance labels (i.e. a bag is positive if at least one of its instances is positive):

````@example musk
y = map(i -> maximum(y[i]) + 1, ds.bags)
y_oh = onehotbatch(y, 1:2)
````

### Model construction

Once the data are in `Mill` internal format, we will manually create a model. [`BagModel`](@ref) is designed to implement a basic multi-instance learning model utilizing two feed-forward networks with an aggregaton operator in between:

````@example musk
model = BagModel(
    Dense(166, 50, Flux.tanh),
    SegmentedMeanMax(50),
    Chain(Dense(100, 50, Flux.tanh), Dense(50, 2)))
````

Instances are first passed through a single layer with 50 neurons (input dimension is 166) with `tanh` non-linearity, then we use `mean` and `max` aggregation functions simultaneously (for some problems, max is better then mean, therefore we use both), and then we use one layer with 50 neurons and `tanh` nonlinearity followed by linear layer with 2 neurons (output dimension). We check that forward pass works

````@example musk
model(ds)
````

!!! ukn "Easier model construction"
    Note that the model can be obtained in a more straightforward way using [Model reflection](@ref).

### Training

Since `Mill` is entirely compatible with [`Flux.jl`](https://fluxml.ai), we can use its `Adam` optimizer:

````@example musk
opt_state = Flux.setup(Adam(), model)
````

...define a loss function as `Flux.logitcrossentropy`:

````@example musk
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
````

...and run a simple training procedure using the `Flux.train!` procedure:

````@example musk
for e in 1:100
    if e % 10 == 1
        @info "Epoch $e" training_loss=loss(model, ds, y_oh)
    end
    Flux.train!(loss, model, [(ds, y_oh)], opt_state)
end
````

Finally, we calculate the (training) accuracy:

````@example musk
mean(Flux.onecold(model(ds), 1:2) .== y)
````

```@setup musk
Pkg.activate(old_path)
```
