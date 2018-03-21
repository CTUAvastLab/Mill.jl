using Revise
using NestedMill
using Base.Test
using Lazy
using Flux

import NestedMill: DictEntry, recommendscheme, accomodate!

samples = open("prescription.jsonl") do fid
	readlines(fid)
end
schema = DictEntry()
foreach(f -> accomodate!(schema,JSON.parse(f)),samples)
reflector = NestedMill.recommendscheme(Float32,schema,2000)


import NestedMill: ExtractCategorical, ExtractBranch
#since npi is rubish, remove it from the extraction
reflector = ExtractBranch(Float32,nothing,reflector.other);

fnames = ["specialty","years_practicing","settlement_type","region"]
vec = Dict(map(k -> (k,ExtractCategorical(Float32,schema["provider_variables"][k])),fnames))
reflector.other["provider_variables"] = ExtractBranch(Float32,vec,nothing)

println(NestedMill.tojson(reflector,2))
ds = @>> samples[1:1000] map(s-> reflector(JSON.parse(s)))
ds = cat(ds...)


import NestedMill: ModelNode

d = map(x -> size(x.data,1),ds.data)
model = ModelNode([Dense(d[1],20,relu),Dense(d[2],20,relu)])
