using Documenter, DocumenterCitations
using Mill

DocMeta.setdocmeta!(Mill, :DocTestSetup, :(using Mill); recursive=true)

makedocs(
         CitationBibliography(joinpath(@__DIR__, "references.bib")),
         sitename = "Mill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  prettyurls=get(ENV, "CI", nothing) == "true",
                                  assets=["assets/favicon.ico", "assets/custom.css"]),
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
                                 "GNNs in 16 lines" => "examples/graphs.md",
                                 "DAGs" => "examples/dag.md",
                                 "Processing JSONs" => "examples/jsons.md"
                                ],
                  "External tools" => [
                                     "HierarchicalUtils.jl" => "tools/hierarchical.md",
                                    ],
                  "References" => "references.md",
                  "Citation" => "citation.md"
                 ],

        )

deploydocs(
           repo = "github.com/pevnak/Mill.jl.git",
          )
