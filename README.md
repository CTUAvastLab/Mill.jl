# Mill


Mill is a library to implement flexible multi-instance learning models as described in *Discriminative models for multi-instance problems with tree-structure, Tomáš Pevný, Petr Somol, 2016* https://arxiv.org/abs/1703.02868 and *Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomáš Pevný, Petr Somol, 2016* https://arxiv.org/abs/1609.07257. An example of how to use the library on simple MUSK problem is in example directory.


## What is Multiple instance learning (MIL) problem?

In the prototypical machine learning problem the input sample `x` is a vector or matrix of a fixed dimension, or a sequence. In MIL problems the sample `x` is a set of vectors (or matrices) `(x_1, x_2, ..., x_n)`, which means that order does not matter, and which is also the feature making MIL problems different from sequences.

Pevny and Somol has proposed simple way to solve MIL problems with neural networks. The network consists from two non-linear layers, with mean (or maximum) operation sandwiched between nonlinearities. Denoting f_1, f_2 layers of neural network, the output is calculated as ``f(x) = f_2 (\frac{1}{n}\sum_{i=1}^{n} f_1(x_i))``. *Approximation capability of neural networks on sets of probability measures and tree-structured data, Tomáš Pevný, Vojtěch Kovařík, 2019* https://openreview.net/forum?id=HklJV3A9Ym have further extended the universal approximation theorem to MIL problems.