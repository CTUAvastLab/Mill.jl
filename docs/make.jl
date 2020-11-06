using Documenter
using Mill

makedocs(
    sitename = "Mill",
    format = Documenter.HTML(),
    modules = [Mill],
    pages = ["Home" => "index.md",
    "Examples" => [
    	"Simple" => "examples/simple.md",
    	"Advanced" => "examples/advanced.md",
		# Suggested by Jan Francu, ?extensions to Graph NN? / Dags?
    ],
    "Architecture of Mill" => [
    	"Overview" => "architecture/overview.md",
    	"ReflectInModel" => "architecture/reflectin.md",
		"Handling strings" => "architecture/strings.md",
    	"Aggregations" => "architecture/aggregation.md",
    	"Missing values" => "architecture/missing.md",
    	"Custom Nodes" => "architecture/custom.md",
	],
    "Helper tools" => [
    	"HierarichalUtils.jl" => "tools/hierarchical.md",
		# Conveniences (Lenses)
		# External Tools (JsonGrinder, HierarchicalUtils)
    ],
	],

)

deploydocs(
    repo = "github.com/pevnak/Mill.jl.git",
)
