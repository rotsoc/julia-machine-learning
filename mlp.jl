using Pkg
for p in ("Knet","AutoGrad","Plots","ProgressMeter")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

# Load data (see 02.mnist.ipynb)
using Knet: Knet, KnetArray, gpu, minibatch
include(Knet.dir("data","mnist.jl"))  # Load data
dtrn,dtst = mnistdata(xsize=(784,:)); # dtrn and dtst = [ (x1,y1), (x2,y2), ... ] where xi,yi are minibatches of 100

# For running experiments
using Knet: SGD, train!, nll, zeroone
import ProgressMeter

function trainresults(file,model; o...)
    if (print("Train from scratch? ");readline()[1]=='y')
        results = Float64[]; updates = 0; prog = ProgressMeter.Progress(60000)
        function callback(J)
            if updates % 600 == 0
                push!(results, nll(model,dtrn), nll(model,dtst), zeroone(model,dtrn), zeroone(model,dtst))
                ProgressMeter.update!(prog, updates)
            end
            return (updates += 1) <= 60000
        end
        Knet.train!(model, dtrn; callback=callback, optimizer=SGD(lr=0.1), o...)
        Knet.save(file,"results",reshape(results, (4,:)))
    end
    isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
    results = Knet.load(file,"results")
    println(minimum(results,dims=2))
    return results
end

using AutoGrad: Param


# Redefine Linear model (See 03.lin.ipynb):
struct Linear; w; b; end
(f::Linear)(x) = (f.w * x .+ f.b)

# Linear(inputsize,outputsize) constructs a randomly initalized Linear model:
Linear(inputsize::Int,outputsize::Int) = Linear(param(outputsize,inputsize),param0(outputsize))
param(d...; init=xavier, atype=atype())=Param(atype(init(d...)))
param0(d...; atype=atype())=param(d...; init=zeros, atype=atype)
xavier(o,i) = (s = sqrt(2/(i+o)); 2s .* rand(o,i) .- s)
atype()=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

# MLP is a bunch of linear layers stuck together:
struct MLP; layers; end

# MLP(x,h1,h2,...,hn,y): constructor for a n hidden layer MLP:
MLP(h::Int...)=MLP(Linear.(h[1:end-1], h[2:end]))

# Here is an example model
model=MLP(784,64,10)
summary.(l.w for l in model.layers)

using Plots; default(fmt=:png,ls=:auto)
ENV["COLUMNS"]=92

# Let us try to just chain multiple linear layers
function (m::MLP)(x)
    for layer in m.layers
        x = layer(x)
    end
    return x
end

# Train a linear model
lin1 = trainresults("lin1.jld2", Linear(784,10)); # 113s [0.241861, 0.266827, 0.0668667, 0.0747]

# Train a multi-linear model
mlp1 = trainresults("mlp1.jld2", MLP(784,64,10)); # 110s [0.242123, 0.285064, 0.0697, 0.0799]

Knet.gc() # to save gpu memory

# Using nonlinearities between layers results in a model with higher capacity and helps underfitting
# relu(x)=max(0,x) is a popular function used for this purpose, it replaces all negative values with zeros.
using Knet: relu

function (m::MLP)(x)
    for layer in m.layers
        x = layer(x)
        x = (layer == m.layers[end] ? x : relu.(x))  # <-- This one line makes a big difference
    end
    return x
end

mlp2 = trainresults("mlp2.jld2", MLP(784,64,10));  # 124s [0.00632429, 0.0888991, 0.00055, 0.0259]

# MLP solves the underfitting problem!  A more serious overfitting problem remains.
plot([lin1[1,:], lin1[2,:], mlp2[1,:], mlp2[2,:]], ylim=(0.0,0.4),
     labels=[:trnLin :tstLin :trnMLP :tstMLP],xlabel="Epochs",ylabel="Loss")

Knet.gc() # to save gpu memory

# Define new loss functions for L1 and L2 regularization
using Knet: nll
nll1(m::MLP,x,y; λ=0, o...) = nll(m,x,y; o...) + λ * sum(sum(abs, l.w) for l in m.layers)
nll2(m::MLP,x,y; λ=0, o...) = nll(m,x,y; o...) + λ * sum(sum(abs2,l.w) for l in m.layers)

mlp3 = trainresults("mlp3.jld2", MLP(784,64,10); loss=nll1, λ=4f-5); # 133s [0.0262292, 0.0792939, 0.0066, 0.0237]

Knet.gc() # to save gpu memory

using Knet: dropout
# Dropout is another way to address overfitting
function (m::MLP)(x; pdrop=0)
    for (i,layer) in enumerate(m.layers)
        p = (i <= length(pdrop) ? pdrop[i] : pdrop[end])
        x = dropout(x, p)     # <-- This one line helps generalization
        x = layer(x)
        x = (layer == m.layers[end] ? x : relu.(x))
    end
    return x
end

mlp4 = trainresults("mlp4.jld2", MLP(784,64,10); pdrop=(0.2,0.0));  # 119s [0.0126026, 0.0648639, 0.0033, 0.0189]
# overfitting less, loss results improve 0.0808 -> 0.0639
plot([mlp2[1,:], mlp2[2,:], mlp4[1,:], mlp4[2,:]], ylim=(0.0,0.15),
     labels=[:trnMLP :tstMLP :trnDrop :tstDrop],xlabel="Epochs",ylabel="Loss")
# this time error also improves 0.0235 -> 0.0188
plot([mlp2[3,:], mlp2[4,:], mlp4[3,:], mlp4[4,:]], ylim=(0.0,0.04),
          labels=[:trnMLP :tstMLP :trnDrop :tstDrop],xlabel="Epochs",ylabel="Error")

Knet.gc() # to save gpu memory
# The current trend is to use models with higher capacity tempered with dropout
mlp = trainresults("mlp.jld2", MLP(784,256,10); pdrop=(0.2,0.0));  # 123s [0.00365709, 0.0473298, 0.000283333, 0.0141]
plot([mlp4[1,:], mlp4[2,:], mlp[1,:], mlp[2,:]],ylim=(0,0.15),
    labels=[:trn64 :tst64 :trn256 :tst256],xlabel="Epochs",ylabel="Loss")

# We are down to 0.015 error.
plot([mlp4[3,:], mlp4[4,:], mlp[3,:], mlp[4,:]],ylim=(0,0.04),
    labels=[:trn64 :tst64 :trn256 :tst256],xlabel="Epochs",ylabel="Error")
