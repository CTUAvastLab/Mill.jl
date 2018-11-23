# Mill


Mill is a library build on top of Flux.jl aimed to prototype flexible multi-instance learning models as described in *Discriminative models for multi-instance problems with tree-structure, Tomáš Pevný, Petr Somol, 2016* https://arxiv.org/abs/1703.02868 and *Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomáš Pevný, Petr Somol, 2016* https://arxiv.org/abs/1609.07257. 


## What is Multiple instance learning (MIL) problem?

In the prototypical machine learning problem the input sample `x` is a vector or matrix of a fixed dimension, or a sequence. In MIL problems the sample `x` is a set of vectors (or matrices) `(x_1, x_2, ..., x_n)`, which means that order does not matter, and which is also the feature making MIL problems different from sequences.

Pevny and Somol has proposed simple way to solve MIL problems with neural networks. The network consists from two non-linear layers, with mean (or maximum) operation sandwiched between nonlinearities. Denoting f_1, f_2 layers of neural network, the output is calculated as ``f(x) = f_2 (\frac{1}{n}\sum_{i=1}^{n} f_1(x_i))``. *Approximation capability of neural networks on sets of probability measures and tree-structured data, Tomáš Pevný, Vojtěch Kovařík, 2019* https://openreview.net/forum?id=HklJV3A9Ym have further extended the universal approximation theorem to MIL problems.

## Multiple instance learning on Musk 1
Musk dataset is a classic problem of the field used in the publication which has given the class of problems its name. *Solving the multiple instance problem with axis-parallel rectangles, Dietterich, Thomas G., Richard H. Lathrop, and Tomás Lozano-Pérez, 1997*

Below is a little walk-through how to solve the problem using Mill library. The full example is shown in `example/musk.jl`

Let's start by importing all libraries
```
using FileIO, JLD2, Flux, MLDataPattern, Mill, Statistics
using Flux: throttle
using Mill: reflectinmodel
```

Loading a dataset from file and folding it in Mill's data-structures is in the following dataset. `musk.jld2` contains matrix with features, `fMat`, that id of sample (called bag in MIL terminology) to which each instance (column in `fMat`) belongs to, and finally a label of each instance in `y`. 
BagNode is a structure which holds feature matrix and ranges of columns of each bag. Note that the library requires instances of each bag to be next to each other. Finally, BagNode can be concatenated (use `catobs`), you can make subset using `getindex`, and the library is compatible with a popular `MLDataPattern` package,
```
function loaddata()
  fMat = load("musk.jld2","fMat");               # matrix with instances, each column is one sample
  bagids = load("musk.jld2","bagids");           # ties instances to bags
  data = BagNode(ArrayNode(fMat), bagids)      # create BagDataset
  y = load("musk.jld2","y");                     # load labels
  y = map(i -> maximum(y[i]) + 1, data.bags)    # create labels on bags
  return(data, y)
end
```


Once we have a data, we can create manually a model. `BagModel` is designed to implement basic multi-instance learning model as described above. Below, we use a simple model, where instances are first passed through a single layer with 10 neurons (input dimension is 166) with `relu` non-linearity, then we use mean and maximum aggregation functions simultaneously (for some problems, max is better then mean, therefore we use both), and then we use one layer with 10 neurons and relu nonlinearity followed by output linear layer with 2 neurons (output dimension).
```
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.relu)),   # model on the level of Flows
    SegmentedMeanMax(),
    ArrayModel(Chain(Dense(20, 10, Flux.relu), Dense(10, 2))))         # model on the level of bags
```

The loss function is classical cross-entropy. Note the use of `getobs` before passing the data to the model. This is artifact of lazy sub-setting  of `MLDataPattern` library
```
loss(x,y) = Flux.logitcrossentropy(model(getobs(x)).data, Flux.onehotbatch(y, 1:2));
```


Finally, we put everything together. The below code should resemble an example from `Flux.jl` library. Note that the library is fully compatible with the training and also note the use of `RandomBatches` from MLDataPattern to train the model for 2000 steps, where each minibatch contains 100 samples.

```
data, y = loaddata()
dataset = RandomBatches((data,y), 100, 2000)
evalcb = () -> @show(loss(data, y))
opt = Flux.ADAM(params(model))
Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))
```
 
Because we did not left any data for validation, we can only calculate accuracy on training data, which should be not so surprisingly 0.

```
mean(Flux.onecold( model(data).data) .!= y)
```


## More complicated models
The main advantage of the Mill library is that it allows to arbitrarily nest and cross-product  BagModels, as is described in Theorem 5 of *Approximation capability of neural networks on sets of probability measures and tree-structured data, Tomáš Pevný, Vojtěch Kovařík, 2019*

Let's start the demonstration by nesting two MIL problems. The outer MIL model contains three samples. The first sample contains another bag (inner MIL) problem with two instances, the second sample contains two inner bags with total of three instances, and finally the third sample contains two inner bags with four instances.
```
julia> ds = BagNode(BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),[1:1,2:3,4:5])
BagNode with 3 bag(s)
  └── BagNode with 5 bag(s)
        └── ArrayNode(4, 10)
```

We can create the model manually as in the case of Musk as
```
julia> m = BagModel(
    BagModel(
        ArrayModel(Dense(4, 3, Flux.relu)),   
        SegmentedMeanMax(),
        ArrayModel(Dense(6, 3, Flux.relu))),
    SegmentedMeanMax(),
    ArrayModel(Chain(Dense(6, 3, Flux.relu), Dense(3,2))))
BagModel
  ├── BagModel
  │     ├── Dense(4, 3, NNlib.relu)
  │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     └── Dense(6, 3, NNlib.relu)
  ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  └── Chain(Dense(6, 3, NNlib.relu), Dense(3, 2))
```
and we can apply the model as
```
julia> m(ds)
ArrayNode(2, 3)
```

Since constructions of large models can be a process prone to errors, there is a function `reflectinmodel` which tries to automatize it keeping track of dimensions. It accepts a first parameter a sample `ds`, the second is a function returning layer (or set of layers) with input dimension `d`, and the third function is a function returning aggregation functions for `BagModel`. Using the function on the above example creates a model
```
julia> m, k = reflectinmodel(ds, d -> Dense(d, 5, relu), d -> SegmentedMeanMax())
BagModel
  ├── BagModel
  │     ├── Dense(4, 5, NNlib.relu)
  │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     └── Dense(10, 5, NNlib.relu)
  ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  └── Dense(10, 5, NNlib.relu)
```
Note that the function returns an output dimension of resulting model. This aimed to facilitate adding the last output linear layer. At this moment, this is slightly cumbersome, as we need to reconstruct the last model node.
```
julia> m = BagModel(m.im, m.a, Chain(Dense(10, 5, relu), Dense(5, 2)))
BagModel
  ├── BagModel
  │     ├── Dense(4, 5, NNlib.relu)
  │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     └── Dense(10, 5, NNlib.relu)
  ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  └── Chain(Dense(10, 5, NNlib.relu), Dense(5, 2))
```

Let's test the model
```
julia> m(ds).data
Tracked 2×3 Array{Float64,2}:
  0.0768481   0.5559    0.111104
 -0.207159   -1.78459  -0.00677607
```

## Even more complicated models
As already mentioned above, the datasets can contain Cartesian products of MIL and normal (non-MIL) problems. Let's do a quick demo.
```
julia> ds = BagNode(
    TreeNode(
        (BagNode(ArrayNode(randn(4,10)),[1:2,3:4,5:5,6:7,8:10]),
        ArrayNode(randn(3,5)),
        BagNode(
            BagNode(ArrayNode(randn(2,30)),[i:i+1 for i in 1:2:30]),
            [1:3,4:6,7:9,10:12,13:15]),
        ArrayNode(randn(2,5)))),
    [1:1,2:3,4:5])

BagNode with 3 bag(s)
  └── TreeNode{4}
        ├── BagNode with 5 bag(s)
        │     └── ArrayNode(4, 10)
        ├── ArrayNode(3, 5)
        ├── BagNode with 5 bag(s)
        │     └── BagNode with 15 bag(s)
        │           └── ArrayNode(2, 30)
        └── ArrayNode(2, 5)
```
For this, we really want to create model automatically despite it being sub-optimal.
```
julia>  m, k = reflectinmodel(ds, d -> Dense(d, 3, relu), d -> SegmentedMeanMax())
BagModel
  ├── ProductModel(
  │     ├── BagModel
  │     │     ├── Dense(4, 3, NNlib.relu)
  │     │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     │     └── Dense(6, 3, NNlib.relu)
  │     ├── Dense(3, 3, NNlib.relu)
  │     ├── BagModel
  │     │     ├── BagModel
  │     │     │     ├── Dense(2, 3, NNlib.relu)
  │     │     │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     │     │     └── Dense(6, 3, NNlib.relu)
  │     │     ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  │     │     └── Dense(6, 3, NNlib.relu)
  │     └── Dense(2, 3, NNlib.relu)
  │   ) ↦  Dense(12, 3, NNlib.relu)
  ├── Aggregation((Mill._segmented_mean, Mill._segmented_max))
  └── Dense(6, 3, NNlib.relu)
```