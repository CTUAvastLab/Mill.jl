```@setup mill 
using Mill
```

# Bag aggregation

A wrapper type `Aggregation` and all subtypes of `AggregationOperator` it wraps are structures that are responsible for mapping of vector representations of multiple instances into a single vector. They all operate element-wise and independently of dimension and thus the output has the same size as representations on the input, unless the [Concatenation](@ref) of multiple operators is used or [Bag Count](@ref) is enabled.

Some setup:

```@repl mill
d = 2
X = Float32.([1 2 3 4; 8 7 6 5])
n = ArrayNode(X)
bags = AlignedBags([1:1, 2:3, 4:4])

Mill.bagcount!(false)
```

Different choice of operator, or their combinations, are suitable for different problems. Nevertheless, because the input is interpreted as an unordered bag of instances, every operator is invariant to permutation and also does not scale when increasing size of the bag.

## Non-parametric aggregation

`SegmentedMax` is the most straightforward operator defined in one dimension as follows:

```math
a_{\max}(\{x_1, \ldots, x_k\}) = \max_{i = 1, \ldots, k} x_i
```

where ``\{x_1, \ldots, x_k\}`` are all instances of the given bag. In `Mill.jl`, the operator is constructed this way:

```@repl mill
a_max = SegmentedMax(d)
```

!!! ukn "Dimension"
    The dimension of input is required so that the default parameters `ψ` can be properly instantiated (see [Missing data](@ref) for details).

The application is straightforward and can be performed on both raw `AbstractArray`s or `ArrayNode`s:

```@repl mill
a_max(X, bags)
a_max(n, bags)
```

Since we have three bags, we have three columns in the output, each storing the maximal element over all instances of the given bag.

### `SegmentedMean`

`SegmentedMean` is defined as:

```math
a_{\operatorname{mean}}(\{x_1, \ldots, x_k\}) = \frac{1}{k} \sum_{i = 1}^{k} x_i
```

and used the same way:

```@repl mill
a_mean = SegmentedMean(d)
a_mean(X, bags)
a_mean(n, bags)
```

!!! ukn "Sufficiency of the mean operator"
    In theory, `SegmentedMean` is sufficient for approximation ([Pevny2019](@cite)), but in practice, a combination of multiple operators performes better.

The max aggregation is suitable for cases when one instance in the bag may give evidence strong enough to predict the label. On the other side of the spectrum lies the mean aggregation function, which detects well trends identifiable globally over the whole bag.

## Parametric aggregation

Whereas non-parametric aggregations do not use any parameter, parametric aggregations represent an entire class of functions parametrized by one or more real vectors of parameters, which can be even learned during training.

### `SegmentedLSE`

`SegmentedLSE` (log-sum-exp) aggregation ([Kraus2015](@cite)) is parametrized by a vector of positive numbers ``\bm{r} \in (\mathbb{R}^+)^d`` m that specifies one real parameter for computation in each output dimension:

```math
a_{\operatorname{lse}}(\{x_1, \ldots, x_k\}; r) = \frac{1}{r}\log \left(\frac{1}{k} \sum_{i = 1}^{k} \exp({r\cdot x_i})\right)
```

With different values of ``r``, LSE behaves differently and in fact both max and mean operators are limiting cases of LSE. If ``r`` is very small, the output approaches simple mean, and on the other hand, if ``r`` is a large number, LSE becomes a smooth approximation of the max function. Naively implementing the definition above may lead to numerical instabilities, however, the `Mill.jl` implementation is numerically stable.

```@repl mill
a_lse = SegmentedLSE(d)
a_lse(X, bags)
```

### `SegmentedPNorm`

(Normalized) ``p``-norm operator ([Gulcehre2014](@cite)) is parametrized by a vector of real numbers ``\bm{p} \in (\mathbb{R}^+)^d``, where ``\forall i \in \{1, \ldots ,m \} \colon p_i \geq 1``, and another vector ``\bm{c} \in (\mathbb{R}^+)^d``. It is computed with formula:

```math
a_{\operatorname{pnorm}}(\{x_1, \ldots, x_k\}; p, c) = \left(\frac{1}{k} \sum_{i = 1}^{k} \vert x_i - c \vert ^ {p} \right)^{\frac{1}{p}}
```

Again, the `Mill.jl` implementation is stable.

```@repl mill
a_pnorm = SegmentedPNorm(d)
a_pnorm(X, bags)
```

Because all parameter constraints are included implicitly (field `\rho` in both types is a real number that undergoes appropriate transformation before being used), both parametric operators are easy to use and do not require any special treatment. Replacing the definition of aggregation operators while constructing a model (either manually or with `reflectinmodel`) is enough.

### Concatenation

To use a concatenation of two or more operators, one can use an `Aggregation` constructor:

```@repl mill
a = Aggregation(a_mean, a_max)
a(X, bags)
```

For the most common combinations, `Mill.jl` provides some convenience definitions:

```@repl mill
SegmentedMeanMax(d)
```

## Weighted aggregation

Sometimes, different instances in the bag are not equally important and contribute to output to a different extent. For instance, this may come in handy when performing importance sampling over very large bags. `SegmentedMean` and `SegmentedPNorm` have definitions taking weights into account:

```math
a_{\operatorname{mean}}(\{(x_i, w_i)\}_{i=1}^k) = \frac{1}{\sum_{i=1}^k w_i} \sum_{i = 1}^{k} w_i \cdot x_i
```

```math
a_{\operatorname{pnorm}}(\{x_i, w_i\}_{i=1}^k; p, c) = \left(\frac{1}{\sum_{i=1}^k w_i} \sum_{i = 1}^{k} w_i\cdot\vert x_i - c \vert ^ {p} \right)^{\frac{1}{p}}
```

This is done in `Mill.jl` by passing an additional parameter:

```@repl mill
w = Float32.([1.0, 0.2, 0.8, 0.5])
a_mean(X, bags, w)
a_pnorm(X, bags, w)
```

For `SegmentedMax` (and `SegmentedLSE`) it is possible to pass in weights, but they are ignored during computation:

```@repl mill
a_max(X, bags, w) == a_max(X, bags)
```

### `WeightedBagNode`

`WeightedBagNode` is used to store instance weights into a dataset. It accepts weights in the constructor:

```@repl mill
wbn = WeightedBagNode(n, bags, w)
```

and passes them to aggregation operators:

```@repl mill
m = reflectinmodel(wbn)
m(wbn)
```

Otherwise, `WeightedBagNode` behaves exactly like the standard `BagNode`.

### Bag count

For some problems, it may be beneficial to use the size of the bag directly and feed it to subsequent layers. Whether this is the case is controlled by `Mill.bagcount!(::Bool)` function. It is on by default, however, it was disabled at the beginning of this section for demonstration purposes. Let's turn it back on:

```@repl mill
Mill.bagcount!(true)
```

In the aggregation phase, bag count appends one more element which stores the bag size to the output after all operators are applied. Furthermore, in `Mill.jl`, we opted to perform a mapping ``x \mapsto \log(x) + 1`` on top of that:

```@repl mill
a_mean(X, bags)
```

The matrix now has three rows, the last one storing the size of the bag.

When the bag count is on, one needs to have a model accepting corresponding sizes:

```@repl mill
bn = BagNode(n, bags)
bm = reflectinmodel(bn)
```

Note that the `bm` (sub)model field of the `BagNode` has size of `(11, 10)`, `10` for aggregation output and `1` for sizes of bags.

```@repl mill
bm(bn)
```

Model reflection takes bag count toggle into account. If we disable it again, `bm` (sub)model has size `(10, 10)`:

```@repl mill
Mill.bagcount!(false)
bm = reflectinmodel(bn)
```

## Default aggregation values

When all aggregation operators are printed, one may notice that all of them store one additional vector `ψ`. This is a vector of default parameters, initialized to all zeros, that are used for empty bags:

```@repl mill
bags = AlignedBags([1:1, 0:-1, 2:3, 0:-1, 4:4])
a_mean(X, bags)
```

See [Missing data](@ref) page for more information.
