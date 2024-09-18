<p align="center">
  <img src="https://github.com/CTUAvastLab/Mill.jl/raw/master/docs/src/assets/logo.svg#gh-light-mode-only" alt="Mill.jl logo"/>
  <img src="https://github.com/CTUAvastLab/Mill.jl/raw/master/docs/src/assets/logo-dark.svg#gh-dark-mode-only" alt="Mill.jl logo"/>
</p>

---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CTUAvastLab/Mill.jl/blob/master/LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://CTUAvastLab.github.io/Mill.jl/stable)
[![Build Status](https://github.com/CTUAvastLab/Mill.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/CTUAvastLab/Mill.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CTUAvastLab/Mill.jl/graph/badge.svg?token=bIjsTgkv8C)](https://codecov.io/gh/CTUAvastLab/Mill.jl)

`Mill.jl` (Multiple Instance Learning Library) is a library aimed to build flexible hierarchical multi-instance learning models built on top of [`Flux.jl`](https://fluxml.ai). It is developed to be:

* **flexible** and **versatile**
* as **general** as possible
* **fast** 
* and dependent on only handful of other packages

[**Watch our introductory talk from JuliaCon 2021** ](https://www.youtube.com/watch?v=Bf0CvltIDbE)

## Installation

Run the following in REPL:

```julia
] add Mill
```

Julia v1.9 or later is required.

## Getting Started

- [Documentation](https://ctuavastlab.github.io/Mill.jl/stable/)
- [API Reference](https://ctuavastlab.github.io/Mill.jl/stable/api/aggregation/)
- [Examples](https://ctuavastlab.github.io/Mill.jl/stable/examples/musk/musk/)

## Citation

Kindly cite our work with the following entries if you find it interesting, please:

* [*JsonGrinder.jl: automated differentiable neural architecture for embedding arbitrary JSON
  data*](https://jmlr.org/papers/v23/21-0174.html)

  ```
  @article{Mandlik2022,
   author = {Šimon Mandlík and Matěj Račinský and Viliam Lisý and Tomáš Pevný},
   issn = {1533-7928},
   issue = {298},
   journal = {Journal of Machine Learning Research},
   pages = {1-5},
   title = {JsonGrinder.jl: automated differentiable neural architecture for embedding arbitrary JSON data},
   volume = {23},
   url = {http://jmlr.org/papers/v23/21-0174.html},
   year = {2022},
  }
  ```

* [*Malicious Internet Entity Detection Using Local Graph
  Inference*](https://ieeexplore.ieee.org/document/10418120) (practical `Mill.jl` application)

  ```
  @article{Mandlik2024,
    author  = {Mandlík, Šimon and Pevný, Tomáš and Šmídl, Václav and Bajer, Lukáš},
    journal = {IEEE Transactions on Information Forensics and Security},
    title   = {Malicious Internet Entity Detection Using Local Graph Inference},
    year    = {2024},
    volume  = {19},
    pages   = {3554-3566},
    doi     = {10.1109/TIFS.2024.3360867}
  }
  ```

* this implementation (fill in the used `version`)

  ```
  @software{Mill,
    author  = {Tomas Pevny and Simon Mandlik},
    title   = {Mill.jl framework: a flexible library for (hierarchical) multi-instance learning},
    url     = {https://github.com/CTUAvastLab/Mill.jl},
    version = {...},
  }
  ```

## Contribution guidelines

If you want to contribute to Mill.jl, be sure to review the
[contribution guidelines](CONTRIBUTING.md).

We use [GitHub issues](https://github.com/CTUAvastLab/Mill.jl/issues) for
tracking requests and bugs.

<a href="https://flyclipart.com/wind-turbine-png-clipart-windmill-pictures-windmill-png-471749">Icon source</a>
