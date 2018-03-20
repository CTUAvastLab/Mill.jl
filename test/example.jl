using Revise
using NestedMill
using Base.Test
using Lazy

import NestedMill: DictEntry, recommendscheme, accomodate!, called
import NestedMill: Categorical, Branch

samples = open("prescription.jsonl") do fid
	readlines(fid)
end
schema = DictEntry()
foreach(f -> accomodate!(schema,JSON.parse(f)),samples)
reflector = NestedMill.recommendscheme(Float32,schema,2000)

#since npi is rubish, remove it from the extraction
reflector = Branch(Float32,nothing,reflector.other);

vec = map(k -> Categorical(sort(collect(keys(schema["provider_variables"][k].counts)))),["specialty","years_practicing","settlement_type","region"])
reflector.other["provider_variables"] = Branch(Float32,vec,nothing)["specialty"] = schema["provider_variables"]["specialty"] 

println(NestedMill.tojson(reflector,2))
ds = @>> samples[1:1000] map(s-> reflector(JSON.parse(s)))
ds = cat(ds...)

