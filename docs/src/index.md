```@raw html
<img class="display-light-only" src="assets/logo.svg" alt="Mill.jl logo" style="width: 40%;"/>
<img class="display-dark-only" src="assets/logo-dark.svg" alt="Mill.jl logo" /style="width: 40%;">
```

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) (Multiple Instance Learning Library) is a library built on top of [`Flux.jl`](https://fluxml.ai) aimed to flexibly prototype *hierarchical multiple instance learning* models as described in [Mandlik2021](@cite), [Pevny2017a](@cite) and  [Pevny2017b](@cite). It is developed to be:

* **flexible** and **versatile**
* as **general** as possible
* **fast** 
* and dependent on only handful of other packages

Watch our [introductory talk](https://www.youtube.com/watch?v=Bf0CvltIDbE) from JuliaCon 2021.

## Installation

Run the following in REPL:

```julia
] add Mill
```

Julia v1.9 or later is required.

## Getting started

For the quickest start, see the [Musk](@ref) example.

* [Motivation](@ref): a brief introduction into the philosophy of `Mill`
* [Manual](@ref Nodes): a brief tutorial into `Mill`
* [Examples](@ref Musk): some examples of `Mill` use
* [External tools](@ref HierarchicalUtils.jl): examples of integration with other packages
* [Public API](@ref Aggregation): extensive API reference
* [References](@ref): related literature
* [Citation](@ref): preferred citation entries
