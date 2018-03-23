using Revise
using NestedMill
using Base.Test
using Lazy
using Flux
using FluxExtensions
using MLDataPattern

import NestedMill: DictEntry, recommendscheme, accomodate!, mapdata, sparsify, reflectinmodel, ExtractCategorical, ExtractBranch

samples = open("prescription.jsonl") do fid
	readlines(fid)
end

#print example of the JSON
JSON.print(JSON.parse(samples[1]),2)

#create schema of the json
schema = DictEntry()
foreach(f -> accomodate!(schema,JSON.parse(f)),samples);

# Create the extractor. Note that we discard NPI, since it is rubbish and also
reflector = NestedMill.recommendscheme(Float32,schema,2000);
extract_data = ExtractBranch(Float32,nothing,reflector.other);
extract_target = ExtractBranch(Float32,nothing,deepcopy(reflector.other));

fnames = ["specialty","years_practicing","settlement_type","region"]
vec = Dict(map(k -> (k,ExtractCategorical(Float32,schema["provider_variables"][k])),fnames))
extract_data.other["provider_variables"] = ExtractBranch(Float32,vec,nothing)

fnames = ["gender"]
vec = Dict(map(k -> (k,ExtractCategorical(Float32,schema["provider_variables"][k])),fnames))
extract_target.other["provider_variables"] = ExtractBranch(Float32,vec,nothing)
delete!(extract_target.other,"cms_prescription_counts")

println(NestedMill.tojson(extract_data,2))
data = @>> samples[1:1000] map(s-> extract_data(JSON.parse(s)));
data = cat(data...);
target = @>> samples[1:1000] map(s-> extract_target(JSON.parse(s)));
target = cat(target...);
target = target.data.data

#make data sparse
# data = mapdata(i -> sparsify(Float32.(i),0.05),data)
data = mapdata(i -> Float32.(i),data)

layerbuilder =  k -> (ResDense(k,10,relu),10)
m,k = reflectinmodel(data, layerbuilder)
m = NestedMill.addlayer(m,Dense(k,2))
m(data)

opt = Flux.Optimise.ADAM(params(m))
fVal = 0.0
for (i,(x,y)) in enumerate(RandomBatches((data,target),100,10000))
	l = FluxExtensions.crossentropy(m(getobs(x)),getobs(y))
	Flux.Tracker.back!(l)
	opt()
	fVal += Flux.data(l)
	if mod(i,100) == 0
		println(i," ",fVal/100)
		fVal = 0.0
	end
end

