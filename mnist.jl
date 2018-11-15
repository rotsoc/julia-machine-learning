using Pkg
for p in ("Knet","Images","ImageMagick")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

# This loads MNIST handwritten digit recognition dataset.
# Traininig and test data go to variables dtrn and dtst
using Knet: Knet, minibatch
include(Knet.dir("data","mnist.jl"))  # defines mnistdata and mnistview
dtrn,dtst = mnistdata(xtype=Array{Float32});

# dtrn contains 600 minibatches of 100 images (total 60000)
# dtst contains 100 minibatches of 100 images (total 10000)
length.((dtrn,dtst))

# Each minibatch is an (x,y) pair where x is 100 28x28x1 images and y contains their labels.
# Here is the first minibatch in the test set:
(x,y) = first(dtst)
summary.((x,y))

# Here is the first five images from the test set:
using Images, ImageMagick
hcat([mnistview(x,i) for i=1:5]...)

# Here are their labels (0x0a=10 is used to represent 0)
@show y[1:5];

# dtrn and dtst are implemented as Julia iterators (see https://docs.julialang.org/en/v1/manual/interfaces)
# This means they can be used in for loops, i.e. `for (x,y) in dtst`
cnt = zeros(Int,10)
for (x,y) in dtst
    for label in y
        cnt[label] += 1
    end
end
@show cnt;
