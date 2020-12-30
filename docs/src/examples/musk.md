```@setup musk
cd("../../example")
using Pkg
Pkg.activate(pwd())
```

```@repl musk
println(pwd())
Pkg.status()
```

# Musk dataset
[`Musk dataset`](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)) is a classic MIL problem of the field, introduced in the problem defining publication [[1](#cit1)]. Below we use to demonstrate, how to solve the problem using Mill.jl. The full example is shown in [example/musk.jl](example/musk.jl), which also contains Julia environment to run it
 
For the demo, we need following libraries and functions.
```julia
julia> using FileIO, JLD2, Statistics, Mill, Flux
julia> using Flux: throttle, @epochs
julia> using Mill: reflectinmodel
julia> using Base.Iterators: repeated
```
The `musk.jld2` contains matrix with features, `fMat`, the id of sample (called bag in MIL terminology) to which each instance (column in `fMat`) belongs to, and the label of each instance in `y` (the label of a bag is a maximum of labels of its instances, i.e. one positive instance in a bag makes itp positive. 
`BagNode` is a structure which holds (i) feature matrix and (ii) ranges identifying which columns in the feature matrix each bag spans. This representation ensures that feed-forward networks do not need to deal with bag boundaries and always process full continuous maatrices. `BagNode` (and generally every `Node`) can be concatenated using `cat` or  `catobs` and indexed using `getindex`.
```julia
julia> fMat = load("musk.jld2", "fMat");         # matrix with instances, each column is one sample
julia> bagids = load("musk.jld2", "bagids");     # ties instances to bags
julia> x = BagNode(ArrayNode(fMat), bagids);     # create BagDataset
julia> y = load("musk.jld2", "y");               # load labels
julia> y = map(i -> maximum(y[i]) + 1, x.bags);  # create labels on bags
julia> y = Flux.onehotbatch(y, 1:2);          # one-hot encoding
```
Once data are in Mill internal format, we can manually create a model. `BagModel` is designed to implement a basic multi-instance learning model utilizing two feed-forward networks with aggregaton layer in between. Below, we use a simple model, where instances are first passed through a single layer with 10 neurons (input dimension is 166) with `relu` non-linearity, then we use `mean` and `max` aggregation functions simultaneously (for some problems, max is better then mean, therefore we use both), and then we use one layer with 10 neurons and `relu` nonlinearity followed by output linear layer with 2 neurons (output dimension).
```julia
julia> model = BagModel(
    ArrayModel(Dense(166, 10, Flux.relu)),                      # model on the level of Flows
    SegmentedMeanMax(10),                                       # aggregation
    ArrayModel(Chain(Dense(20, 10, Flux.relu), Dense(10, 2))))  # model on the level of bags
    
BagModel ↦ ⟨SegmentedMean(10), SegmentedMax(10)⟩ ↦ ArrayModel(Chain(Dense(20, 10, relu), Dense(10, 2)))
  └── ArrayModel(Dense(166, 10, relu))
```

Since Mill is made maximally compatible with `Flux`, we can use its `cross-entropy`
```julia
julia> loss(x, y) = Flux.logitcrossentropy(model(x).data, y);
```
and train it using its tooling
 ```julia
julia> evalcb = () -> @show(loss(x, y));
julia> opt = Flux.ADAM();
julia> @epochs 10 Flux.train!(loss, params(model), repeated((x, y), 1000), opt, cb=throttle(evalcb, 1))

[ Info: Epoch 1
loss(x, y) = 87.793724f0
[ Info: Epoch 2
loss(x, y) = 4.3207192f0
[ Info: Epoch 3
loss(x, y) = 4.2778687f0
[ Info: Epoch 4
loss(x, y) = 0.662226f0
[ Info: Epoch 5
loss(x, y) = 5.76351f-6
[ Info: Epoch 6
loss(x, y) = 3.8146973f-6
[ Info: Epoch 7
loss(x, y) = 2.8195589f-6
[ Info: Epoch 8
loss(x, y) = 2.4878461f-6
[ Info: Epoch 9
loss(x, y) = 2.1561332f-6
[ Info: Epoch 10
loss(x, y) = 1.7414923f-6
```
 
Because I was lazy and I have not left any data for validation, we can only calculate error on the training data, which should be not so surprisingly low.
 ```julia
mean(mapslices(argmax, model(x).data, dims=1)' .!= y)

0.0
```

 <a name="cit1"><b>1</b></a> *Solving the multiple instance problem with axis-parallel rectangles, Dietterich, Thomas G., Richard H. Lathrop, and Tomás Lozano-Pérez, 1997*
