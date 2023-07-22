using Pkg
using Documenter, DocumenterCitations, Literate
using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils

#=
Useful resources for writing docs:
    Julia guidelines: https://docs.julialang.org/en/v1/manual/documentation/
    Documenter syntax: https://juliadocs.github.io/Documenter.jl/stable/man/syntax/ 
    Showcase: https://juliadocs.github.io/Documenter.jl/stable/showcase/
    Doctests: https://juliadocs.github.io/Documenter.jl/stable/man/doctests/

To locally browse the docs, use

python3 -m http.server --bind localhost

in the build directory.

or

julia -e 'using LiveServer; serve(dir="build")'
=#

musk_path = joinpath(@__DIR__, "src", "examples", "musk")
Pkg.activate(musk_path) do
    Pkg.update()
    Pkg.instantiate()

    add_setup(s) = """
    ```@setup musk
    using Pkg
    old_path = Pkg.project().path
    Pkg.activate(pwd())
    Pkg.instantiate()

    ENV["LINES"] = 25
    ENV["COLUMNS"] = 125
    ```
    """ * s * """
    ```@setup musk
    Pkg.activate(old_path)
    ```
    """
    Literate.markdown(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk",
                                                                credit=false, postprocess=add_setup)
    Literate.script(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk", credit=false)
    Literate.notebook(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk")
end

DocMeta.setdocmeta!(Mill, :DocTestSetup, quote
    using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils
    ENV["LINES"] = ENV["COLUMNS"] = typemax(Int)
end; recursive=true)

makedocs(
         CitationBibliography(joinpath(@__DIR__, "references.bib"), style=:numeric),
         sitename = "Mill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  collapselevel=2,
                                  assets=["assets/favicon.ico", "assets/custom.css"]),
         strict = [:eval_block, :example_block, :meta_block, :setup_block],
         modules = [Mill],
         pages = [
                  "Home" => "index.md",
                  "Motivation" => "motivation.md",
                  "Manual" => [
                               "Nodes" => "manual/nodes.md",
                               "More on nodes" => "manual/more_on_nodes.md",
                               "Model reflection" => "manual/reflectin.md",
                               "Bag aggregation" => "manual/aggregation.md",
                               "Data in leaves" => "manual/leaf_data.md",
                               "Missing data" => "manual/missing.md",
                               "Custom nodes" => "manual/custom.md"
                              ],
                  "Examples" => [
                                 "Musk" => "examples/musk/musk.md",
                                 "GNNs in 16 lines" => "examples/gnn.md",
                                 "DAGs" => "examples/dag.md",
                                 "Processing JSONs" => "examples/jsons.md"
                                ],
                  "External tools" => [
                                       "HierarchicalUtils.jl" => "tools/hierarchical.md"
                                      ],
                  "Public API" => [
                            "Aggregation" => "api/aggregation.md",
                            "Bags" => "api/bags.md",
                            "Data nodes" => "api/data_nodes.md",
                            "Model nodes" => "api/model_nodes.md",
                            "Special Arrays" => "api/special_arrays.md",
                            "Switches" => "api/switches.md",
                            "Utilities" => "api/utilities.md"
                           ],
                  "References" => "references.md",
                  "Citation" => "citation.md"
                 ],
        )

deploydocs(
    repo = "github.com/CTUAvastLab/Mill.jl.git"
)
