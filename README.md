<p align="center">
  <img src="https://github.com/CTUAvastLab/Mill.jl/raw/master/docs/src/assets/logo.svg#gh-light-mode-only" alt="Mill.jl logo"/>
  <img src="https://github.com/CTUAvastLab/Mill.jl/raw/master/docs/src/assets/logo-dark.svg#gh-dark-mode-only" alt="Mill.jl logo"/>
</p>

---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CTUAvastLab/Mill.jl/blob/master/LICENSE)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://CTUAvastLab.github.io/Mill.jl/stable)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://CTUAvastLab.github.io/Mill.jl/dev)
[![Build Status](https://github.com/CTUAvastLab/Mill.jl/workflows/CI/badge.svg)](https://github.com/CTUAvastLab/Mill.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/CTUAvastLab/Mill.jl/badge.svg?branch=master)](https://coveralls.io/github/CTUAvastLab/Mill.jl?branch=master)
[![codecov.io](http://codecov.io/github/CTUAvastLab/Mill.jl/coverage.svg?branch=master)](http://codecov.io/github/CTUAvastLab/Mill.jl?branch=master)

`Mill.jl` (Multiple Instance Learning Library) is a library build on top of [`Flux.jl`](https://fluxml.ai) aimed to prototype flexible multi-instance learning models. It is developed to be:

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

Julia v1.6 or later is required.

## Getting Started

- [Documentation](https://ctuavastlab.github.io/Mill.jl/stable/)
- [API Reference](https://ctuavastlab.github.io/Mill.jl/stable/api/aggregation/)
- [Musk Example](https://ctuavastlab.github.io/Mill.jl/stable/examples/musk/musk/)

## Citation

For citing, please use the following entry for the [original paper](https://arxiv.org/abs/2105.09107):
```
@misc{mandlik2021milljl,
      title={Mill.jl and JsonGrinder.jl: automated differentiable feature extraction for learning from raw JSON data}, 
      author={Simon Mandlik and Matej Racinsky and Viliam Lisy and Tomas Pevny},
      year={2021},
      eprint={2105.09107},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

and the following for this implementation (fill in the used `version`):
```
@software{mill2018,
  author = {Tomas Pevny and Simon Mandlik},
  title = {Mill.jl framework: a flexible library for (hierarchical) multi-instance learning},
  url = {https://github.com/CTUAvastLab/Mill.jl},
  version = {...},
}
```

<a href="https://flyclipart.com/wind-turbine-png-clipart-windmill-pictures-windmill-png-471749">Icon source</a>

## Contribution guidelines

If you want to contribute to Mill.jl, be sure to review the
[contribution guidelines](CONTRIBUTING.md).

We use [GitHub issues](https://github.com/CTUAvastLab/Mill.jl/issues) for
tracking requests and bugs.
