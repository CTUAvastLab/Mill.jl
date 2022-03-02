```@setup aggregation 
using Mill, Flux
```

# Bag aggregation

Aggregation operators in [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) are all subtypes of [`AbstractAggregation`](@ref). These structures are responsible for mapping of vector representations of multiple instances into a single vector. They all operate element-wise and independently of dimension and thus the output has the same size as representations on the input, unless the [Concatenation](@ref) of multiple operators is used.

Some setup:

```@repl aggregation
d = 2
X = Float32.([1 2 3 4; 8 7 6 5])
bags = AlignedBags([1:1, 2:3, 4:4])
```

Different choice of operator, or their combinations, are suitable for different problems. Nevertheless, because the input is interpreted as an unordered bag of instances, every operator is invariant to permutation and also does not scale when increasing size of the bag.

## Non-parametric aggregation

### Max aggregation

[`SegmentedMax`](@ref) implements a simple `max` and is the most straightforward operator defined in one dimension as follows:

```math
a_{\max}(\{x_1, \ldots, x_k\}) = \max_{i = 1, \ldots, k} x_i
```

where ``\{x_1, \ldots, x_k\}`` are all instances of the given bag. In `Mill`, the operator is constructed this way:

```@repl aggregation
a_max = SegmentedMax(d)
```

The application is straightforward:

```@repl aggregation
a_max(X, bags)
```

Since we have three bags, we have three columns in the output, each storing the maximal element over all instances of the given bag.

### Mean aggregation

[`SegmentedMean`](@ref) implements `mean` function, defined as:

```math
a_{\operatorname{mean}}(\{x_1, \ldots, x_k\}) = \frac{1}{k} \sum_{i = 1}^{k} x_i
```

and used the same way:

```@repl aggregation
a_mean = SegmentedMean(d)
a_mean(X, bags)
```

!!! ukn "Sufficiency of the mean operator"
    In theory, `mean` aggregation is sufficient for approximation as proven in [Pevny2019](@cite), but in practice, a combination of multiple operators performes better.

The `max` aggregation is suitable for cases when one instance in the bag may give evidence strong enough to predict the label. On the other side of the spectrum lies the mean aggregation function, which detects well trends identifiable globally over the whole bag.

### Sum aggregation

The last non-parametric operator is [`SegmentedSum`](@ref), defined as:

```math
a_{\operatorname{mean}}(\{x_1, \ldots, x_k\}) = \sum_{i = 1}^{k} x_i
```

and used the same way:

```@repl aggregation
a_sum = SegmentedSum(d)
a_sum(X, bags)
```
## Parametric aggregation

Whereas non-parametric aggregations do not use any parameter, parametric aggregations represent an entire class of functions parametrized by one or more real vectors of parameters, which can be even learned during training.

### Log-sum-exp (LSE) aggregation

[`SegmentedLSE`](@ref) (log-sum-exp) aggregation ([Kraus2015](@cite)) is parametrized by a vector of positive numbers ``\bm{r} \in (\mathbb{R}^+)^d`` m that specifies one real parameter for computation in each output dimension:

```math
a_{\operatorname{lse}}(\{x_1, \ldots, x_k\}; r) = \frac{1}{r}\log \left(\frac{1}{k} \sum_{i = 1}^{k} \exp({r\cdot x_i})\right)
```

With different values of ``r``, LSE behaves differently and in fact both max and mean operators are limiting cases of LSE. If ``r`` is very small, the output approaches simple mean, and on the other hand, if ``r`` is a large number, LSE becomes a smooth approximation of the max function. Naively implementing the definition above may lead to numerical instabilities, however, the `Mill` implementation is numerically stable.

```@repl aggregation
a_lse = SegmentedLSE(d)
a_lse(X, bags)
```

### ``p``-norm aggregation

(Normalized) ``p``-norm operator ([Gulcehre2014](@cite)) is parametrized by a vector of real numbers ``\bm{p} \in (\mathbb{R}^+)^d``, where ``\forall i \in \{1, \ldots ,m \} \colon p_i \geq 1``, and another vector ``\bm{c} \in (\mathbb{R}^+)^d``. It is computed with formula:

```math
a_{\operatorname{pnorm}}(\{x_1, \ldots, x_k\}; p, c) = \left(\frac{1}{k} \sum_{i = 1}^{k} \vert x_i - c \vert ^ {p} \right)^{\frac{1}{p}}
```

Again, the `Mill` implementation is stable.

```@repl aggregation
a_pnorm = SegmentedPNorm(d)
a_pnorm(X, bags)
```

Because all parameter constraints are included implicitly (field `ρ` in both types is a real number that undergoes appropriate transformation before being used), both parametric operators are easy to use and do not require any special treatment. Replacing the definition of aggregation operators while constructing a model (either manually or with [`reflectinmodel`](@ref)) is enough.

### Concatenation

To use a concatenation of two or more operators, one can use the [`AggregationStack`](@ref) constructor:

```@repl aggregation
a = AggregationStack(a_mean, a_max)
a(X, bags)
```

For the most common combinations, `Mill` provides some convenience definitions:

```@repl aggregation
SegmentedMeanMax(d)
SegmentedPNormLSE(d)
```

## Weighted aggregation

Sometimes, different instances in the bag are not equally important and contribute to output to a different extent. For instance, this may come in handy when performing importance sampling over very large bags. [`SegmentedMean`](@ref) and [`SegmentedPNorm`](@ref) have definitions taking weights into account:

```math
a_{\operatorname{mean}}(\{(x_i, w_i)\}_{i=1}^k) = \frac{1}{\sum_{i=1}^k w_i} \sum_{i = 1}^{k} w_i \cdot x_i
```

```math
a_{\operatorname{pnorm}}(\{x_i, w_i\}_{i=1}^k; p, c) = \left(\frac{1}{\sum_{i=1}^k w_i} \sum_{i = 1}^{k} w_i\cdot\vert x_i - c \vert ^ {p} \right)^{\frac{1}{p}}
```

This is done in `Mill` by passing an additional parameter:

```@repl aggregation
w = Float32.([1.0, 0.2, 0.8, 0.5])
a_mean(X, bags, w)
a_pnorm(X, bags, w)
```

For [`SegmentedMax`](@ref) and [`SegmentedLSE`](@ref) it is possible to pass in weights, but they are ignored during computation:

```@repl aggregation
a_max(X, bags, w) == a_max(X, bags)
```

### Weighted nodes

[`WeightedBagNode`](@ref) is used to store instance weights into a dataset. It accepts weights in the constructor:

```@repl aggregation
wbn = WeightedBagNode(X, bags, w)
```

and passes them to aggregation operators:

```@repl aggregation
m = reflectinmodel(wbn, d -> Dense(d, 3))
m(wbn)
```

Otherwise, [`WeightedBagNode`](@ref) behaves exactly like the standard [`BagNode`](@ref).

### Bag count

For some problems, it may be beneficial to use the size of the bag directly and feed it to subsequent layers. To do this, wrap an instance of [`AbstractAggregation`](@ref) or [`AggregationStack`](@ref) in the [`BagCount`](@ref) type.

In the aggregation phase, bag count appends one more element which stores the bag size to the output after all operators are applied. Furthermore, `Mill`, performs a mapping ``x \mapsto \log(x) + 1`` on top of that:

```@repl aggregation
a_mean_bc = BagCount(a_mean)
a_mean_bc(X, bags)
```

The matrix now has three rows, the last one storing the size of the bag.

[Model reflection](@ref) adds [`BagCount`](@ref) after each aggregation operator by default.

```@repl aggregation
bn = BagNode(X, bags)
bm = reflectinmodel(bn, d -> Dense(d, 3))
```

Note that the `bm` (sub)model field of the [`BagNode`](@ref) has size of `(7, 3)`, `3` for each of two aggregation outputs and `1` for sizes of bags.

```@repl aggregation
bm(bn)
```

## Default aggregation values

When all aggregation operators are printed, one may notice that all of them store one additional vector `ψ`. This is a vector of default parameters, initialized to all zeros, that are used for empty bags:

```@repl aggregation
bags = AlignedBags([1:1, 0:-1, 2:3, 0:-1, 4:4])
a_mean(X, bags)
```

That's why the dimension of input is required in the constructor. See [Missing data](@ref) page for more information.
