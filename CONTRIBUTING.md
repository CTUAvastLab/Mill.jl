# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you do the following:

- Check if your changes are consistent with the
    [guidelines](#general-guidelines-and-philosophy-for-contribution).
- Check [Julia Ecosystem contributing guide](https://julialang.org/contribute/).
- Changes are consistent with the [Coding Style](https://docs.julialang.org/en/v1/manual/style-guide/).
- Run the [unit tests](#running-unit-tests).

## Good First Issues

While there are not many right now, we do have a section for 
["good for issues"](https://github.com/CTUAvastLab/Mill.jl/labels/good%20first%20issue). 
As mentioned above, if any of these seem interesting but there is no clear next step in your mind, 
please feel free to ask for a suggested step. 
Often times in open source, issues labeled as "good first issue" actually take some back and forth between maintainers 
and contributors before the issues is ready to be tackled by a new contributor.

## How to become a contributor and submit your own code

### Contributing code

If you have improvements to Mill.jl, send us your pull requests! For those
just getting started, Github has a [how to](https://help.github.com/articles/using-pull-requests/).

If you want to contribute, start working through the codebase, navigate to the
[Github "issues" tab](https://github.com/CTUAvastLab/Mill.jl/issues) and start
looking through interesting issues. If you are not sure of where to start, then
start by trying one of the smaller/easier issues here i.e.
[issues with the "good first issue" label](https://github.com/CTUAvastLab/Mill.jl/labels/good%20first%20issue)
and then take a look at the
[issues with the "contributions welcome" label](https://github.com/CTUAvastLab/Mill.jl/labels/stat%3Acontributions%20welcome).
Sometimes these are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for [review](https://github.com/CTUAvastLab/Mill.jl/pulls),
make sure your changes are consistent with the guidelines and follow the recommended Julia coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   When you contribute a new feature to Mill.jl, the maintenance burden is
    (by default) transferred to the CTUAvastLab. This means that the benefit
    of the contribution must be compared against the cost of maintaining the
    feature.

#### Running unit tests

There are two ways to run unit tests.

1. From the CLI in the root directory of the project:
   
    ```bash
    julia --color=yes --project=@. -e 'using Pkg; pkg"test"'
    ```

2. Running the tests from REPL
 
    Make sure you're in the root directory of the project.
    Run
    ```julia
    using Pkg; pkg"test"
    ```

This will run both the unit tests and the doctests for the documentation.
