# Mill.jl
 Mill.jl is a library build on top of `Flux.jl` aimed to flexibly prototype *hierarchical multi-instance learning* models as described in [[1](#cit1)] and  [[2](#cit2)]

## What is Multiple instance learning (MIL) problem?

Why should I care about MIL problems?
Since the seminal paper of Ronald Fisher, the majority of machine learning problems deals with a problem shown below, 

![mnist_preview](assets/iris.svg)

where the input sample ``x`` is a vector (or more generally a tensor) of a fixed dimension, alterantivelly a sequence. 

The consequence is that if we want to classify the iris aboce, and we want to use features describing its leafs, blossoms, etc, we will have a hard time, since every flower has different number of them. This implies that to use the usual "fix dimension" paradigm, we have to either use features from a single flower and single leaf, or aggregate the description of their set, such that the output has a fixed dimension. This is clearly undesirable, as we would like a framework that can flexibly and automatically deals with these nuisances. 


In Multiple instance learning the sample ``x`` is a set of vectors (or matrices) ``\{x_1,\ldots,x_l\}`` with ``x_i \in R^d``, which means that order does not matter, and which is also the feature making MIL problems different from sequences. The multi-instance problems have been introduced in by Tom Diettrich in [[4](#cit4)] in 1997, and extended and generalized in a series of works [[1](#cit1)], [[2](#cit2)], [[3](#cit3)]. The most comprehensive introduction known to authors is [[6](#cit6)]

The **Hierarchical Multiple instance learning** would approach the problem of iris classification as outlined below.

![mnist_preview](assets/iris2.svg)

It will describe each leafs by a vector implying that all leaves are described bu a set of vectors. The same will be done for blossoms. Note that such description allows each flower to have a different numbers of each entity. Finally, there will be a single vector describing a stem, since there is only one.

How does the MIL copes with variability in number of flowers and leafs (in MIL parlance they are called instances and their set is called a bag)? For each MIL problem, there are two feed-forward neural networks with element-wise aggregation operator like `mean` (or `maximum`) sandwiched between them. Denoting those feed-forward networks (FFN) by  ``f_1``  and ``f_2``, the output of a bag calculated is calculated as ``f_2 \left\(\frac{1}{l}\sum_{i=1}^l f_1(x_i) \right\)``, where we have used `mean` as an aggregation function. In [[3](#cit3)], authors have further extended the universal approximation theorem to MIL problems, their Cartesian products, and nested  MIL problems, i.e. a case where instances of one bag are in fact bags. 

This means that the flower in the above Iris example would be described by one bag describing leafs, another bag describing blossoms, and a vector describing stem. The HMIL model would have two FFNs to convert set of leafs to a single vector, another set of two FFNs to convert set of blossoms to a single vector. These two outputs would be concatenated with a description of a stem, which would be fed to yet another FFN providing the final classifications. And since whole scheme is differentiable, we can use standard SGD to optimize all FFNs together using only labels on the level of output.

The Mill library simplifies implementation of machine learning problems with (H)MIL representation. In theory, it can represent any problem that can be written represented in JSONs. That is why we have created a separate tool, JsonGrinder, which helps to Mill your JSONs.


## Relation to Graph Neural Networks
HMIL problems can be seen as a special subset of general graphs. They differ in two important ways
* In general graphs, vertices are of a small number of semantic type, whereas in HMIL problems, the number of semantic types of vertices is much higher (it is helpful to think about HMIL problems as about those for which JSON is a natural representation).
* The computational graph of HMIL is a **tree**, which implies that there exist an efficient inference. Contrary, in general graphs (with loops) there is no efficient inference and one has to resort to message passing (Loopy belief propagation).
* One update message in **loopy belief propagation** can be viewed as a MIL problem, as it has to produce a vector based on infomation inthe neighborhood, which can contain arbitrary number of vertices.

A more detailed overview of this subject can be found in [[6](#cit6)].


## References
 <a name="cit1"><b>1</b></a> *Discriminative models for multi-instance problems with tree-structure, Tomáš Pevný, Petr Somol, 2016*, https://arxiv.org/abs/1703.02868
 
 <a name="cit2"><b>2</b></a> *Using Neural Network Formalism to Solve Multiple-Instance Problems, Tomáš Pevný, Petr Somol, 2016*, https://arxiv.org/abs/1609.07257. 
 
 <a name="cit3"><b>3</b></a> *Approximation capability of neural networks on sets of probability measures and tree-structured data, Tomáš Pevný, Vojtěch Kovařík, 2019*, https://openreview.net/forum?id=HklJV3A9Ym
 
 <a name="cit4"><b>4</b></a> *Solving the multiple instance problem with axis-parallel rectangles, Dietterich, Thomas G., Richard H. Lathrop, and Tomás Lozano-Pérez, 1997*

 <a name="cit5"><b>5</b></a> *Deep sets, Zaheer, Manzil, et al., 2017*,

 <a name="cit6"><b>6</b></a> Simon Mandlik's diploma thesis, 2019*,

