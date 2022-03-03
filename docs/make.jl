using Pkg
using Documenter, DocumenterCitations, Literate
using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils

#=
Useful resources for writing docs:
    Julia guidelines: https://docs.julialang.org/en/v1/manual/documentation/
    Documenter syntax: https://juliadocs.github.io/Documenter.jl/stable/man/syntax/ 
    Showcase: https://juliadocs.github.io/Documenter.jl/stable/showcase/
    Doctests: https://juliadocs.github.io/Documenter.jl/stable/man/doctests/
=#

musk_path = joinpath(@__DIR__, "src", "examples", "musk")
Pkg.activate(musk_path) do
    Pkg.update("Mill")
    Pkg.instantiate()
    Literate.script(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk", credit=false)
    Literate.markdown(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk", credit=false)
    Literate.notebook(joinpath(musk_path, "musk_literate.jl"), musk_path, name="musk")
end

Pkg.update("Mill")
Pkg.instantiate()

DocMeta.setdocmeta!(Mill, :DocTestSetup, quote
    using Mill, Flux, Random, SparseArrays, Setfield, HierarchicalUtils
end; recursive=true)

const STRICT_CHECKS = [:eval_block, :example_block, :meta_block, :setup_block]

makedocs(
         CitationBibliography(joinpath(@__DIR__, "references.bib")),
         sitename = "Mill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  collapselevel = 2,
                                  prettyurls=get(ENV, "CI", nothing) == "true",
                                  assets=["assets/favicon.ico", "assets/custom.css"]),
         strict = STRICT_CHECKS,
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
