using Pkg
for p in ("Knet","AutoGrad","Plots","Images","ImageMagick","ProgressMeter")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

# Load data (see 02.mnist.ipynb)
using Knet: Knet, minibatch
include(Knet.dir("data","mnist.jl"))
dtrn,dtst = mnistdata(xsize=(784,:),xtype=Array{Float32});

# We will use a callable object to define our linear model
struct Linear; w; b; end
(model::Linear)(x) = model.w * x .+ model.b

# Let's take the first minibatch from the test set
x,y = first(dtst)
summary.((x,y))

# Initialize a random Linear model
model = Linear(randn(10,784)*0.01, zeros(10))

# Display its prediction on the first minibatch: a 10xN score matrix
ENV["COLUMNS"]=72
ypred = model(x)

# correct answers are given as an array of integers
y'

# We can calculate the accuracy of our model for the first minibatch
using Statistics
accuracy(model,x,y) = mean(y' .== map(i->i[1], findmax(Array(model(x)),dims=1)[2]))
accuracy(model,x,y)

# We can calculate the accuracy of our model for the whole test set
using Knet: Data  # type of dtrn and dtst
accuracy(model,data::Data) = mean(accuracy(model,x,y) for (x,y) in data)
accuracy(model,dtst)

# ZeroOne loss (or error) is defined as 1 - accuracy
zeroone(x...) = 1 - accuracy(x...)
zeroone(model,dtst)

# Negative log likelihood (aka cross entropy, softmax loss, NLL)
function nll(model, x, y)
    scores = model(x)
    expscores = exp.(scores)
    probabilities = expscores ./ sum(expscores, dims=1)
    answerprobs = (probabilities[y[i],i] for i in 1:length(y))
    mean(-log.(answerprobs))
end

# Calculate NLL of our model for the first minibatch
nll(model,x,y)

# per-instance average negative log likelihood for the whole test set
nll(model,data::Data) = mean(nll(model,x,y) for (x,y) in data)
nll(model,dtst)

using AutoGrad
@doc AutoGrad

using Random
Random.seed!(9);

# To compute gradients we need to mark fields of f as Params:
model = Linear(Param(randn(10,784)), Param(zeros(10)))

# We can still do predictions with f and calculate loss:
nll(model,x,y)

# And we can do the same loss calculation also computing gradients:
J = @diff nll(model,x,y)

# To get the actual loss value from J:
value(J)

# To get the gradient of a parameter from J:
grdw = grad(J,model.w)

# Note that each gradient has the same size and shape as the corresponding parameter:
grdb = grad(J,model.b)

# Meaning of gradient: If I move the last entry of f.b by epsilon, loss will go up by 0.792576 epsilon!
@show grdb;

@show model.b;

nll(model,x,y)     # loss for the first minibatch with the original parameters

model.b[10] = 0.1   # to numerically check the gradient let's move the last entry of f.b by +0.1.
@show model.b;

nll(model,x,y)     # We see that the loss moves by ≈ +0.79*0.1 as expected.

model.b[10] = 0

# Without AutoGrad we would have to define the gradients manually:
function nllgrad(model,x,y)
    scores = model(x)
    expscores = exp.(scores)
    probabilities = expscores ./ sum(expscores, dims=1)
    for i in 1:length(y); probabilities[y[i],i] -= 1; end
    dJds = probabilities / length(y)
    dJdw = dJds * x'
    dJdb = vec(sum(dJds,dims=2))
    dJdw,dJdb
end;

grdw2, grdb2 = nllgrad(model,x,y)

using LinearAlgebra: axpy! # axpy!(x,y) sets y[:]=a*x+y

function train!(model, data)
    for (x,y) in data
        loss = @diff Knet.nll(model,x,y)  # Knet.nll is bit more efficient
        for param in (model.w, model.b)
            ∇param = grad(loss, param)
            axpy!(-0.1, ∇param, value(param))
        end
    end
end

# Let's try a randomly initialized model for 10 epochs
model = Linear(Param(randn(10,784)*0.01), Param(zeros(10)))
dtrn.xtype = dtst.xtype = Array{Float32}
@show nll(model,dtst)
@time for i=1:10; train!(model,dtrn); end # 17s
@show nll(model,dtst)

# To work on the GPU, all we have to do is convert our Arrays to KnetArrays:
using Knet: KnetArray   # KnetArrays are allocated on and operated by the GPUs
if Knet.gpu() >= 0      # Knet.gpu() returns a device id >= 0 if there is a GPU, -1 otherwise
    ka = KnetArray{Float32}
    dtrn.xtype = dtst.xtype = ka
    model = Linear(Param(ka(randn(10,784)*0.01)), Param(ka(zeros(10))))
    @show nll(model,dtst)
    @time for i=1:10; train!(model,dtrn); end # 7.8s
    @show nll(model,dtst)
end

# Let's collect some data to draw training curves and visualizing weights:
using ProgressMeter: @showprogress

function trainresults(file, epochs)
    results = []
    pa(x) = Knet.gpu() >= 0 ? Param(KnetArray{Float32}(x)) : Param(Array{Float32}(x))
    model = Linear(pa(randn(10,784)*0.01), pa(zeros(10)))
    @showprogress for epoch in 1:epochs  # 100ep 77s (0.2668, 0.0744)
        push!(results, deepcopy(model), Knet.nll(model,dtrn), Knet.nll(model,dtst), zeroone(model,dtrn), zeroone(model,dtst))
        train!(model,dtrn)
    end
    results = reshape(results, (5, :))
    Knet.save(file,"results",results)
end

# Use Knet.load and Knet.save to store models, results, etc.
if (print("Train from scratch? (~77s) "); readline()[1]=='y')
    trainresults("lin.jld2",100)  # (0.2668679f0, 0.0745)
end
isfile("lin.jld2") || download("http://people.csail.mit.edu/deniz/models/tutorial/lin.jld2","lin.jld2")
lin = Knet.load("lin.jld2","results")
minimum(lin[3,:]), minimum(lin[5,:])

using Plots; # default(fmt = :png)
# Demonstrates underfitting: training loss not close to 0
# Also slight overfitting: test loss higher than train
plot([lin[2,:], lin[3,:]],ylim=(.0,.4),labels=[:trnloss :tstloss],xlabel="Epochs",ylabel="Loss")
# this is the error plot, we get to about 7.5% test error, i.e. 92.5% accuracy
plot([lin[4,:], lin[5,:]],ylim=(.0,.12),labels=[:trnerr :tsterr],xlabel="Epochs",ylabel="Error")

# Let us visualize the evolution of the weight matrix as images below
# Each row is turned into a 28x28 image with positive weights light and negative weights dark gray
using Images, ImageMagick #, IJulia
for t in 10 .^ range(0,stop=log10(size(lin,2)),length=10) #logspace(0,2,20)
    i = floor(Int,t)
    f = lin[1,i]
    w1 = reshape(Array(value(f.w))', (28,28,1,10))
    w2 = clamp.(w1.+0.5,0,1)
    # IJulia.clear_output(true)
    display(hcat([mnistview(w2,i) for i=1:10]...))
    display("Epoch $i")
    sleep(1) # (0.96^i)
end
