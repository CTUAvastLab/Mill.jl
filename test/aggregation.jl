using Flux
import Mill: segmented_mean, segmented_max, segmented_meanmax
import Flux.Tracker: gradcheck

gradcheck(x -> sum(segmented_mean(x,[1:4])),randn(4,4))
gradcheck(x -> sum(segmented_mean(x,[1:4,5:5,6:8])),randn(4,8))
gradcheck(x -> sum(segmented_max(x,[1:4])),randn(4,4))
gradcheck(x -> sum(segmented_max(x,[1:4,5:5,6:8])),randn(4,8))
gradcheck(x -> sum(segmented_meanmax(x,[1:4])),randn(4,4))
gradcheck(x -> sum(segmented_meanmax(x,[1:4,5:5,6:8])),randn(4,8))