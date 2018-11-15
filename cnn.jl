# Convolutional Neural Networks
using Pkg
for p in ("Knet","Plots","ProgressMeter")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

# Convolution operator in Knet
using Knet: conv4
@doc conv4

# Convolution in 1-D
@show w = reshape([1.0,2.0,3.0], (3,1,1,1))
@show x = reshape([1.0:7.0...], (7,1,1,1))
@show y = conv4(w, x);  # size Y = X - W + 1 = 5 by default

# Padding
@show y2 = conv4(w, x, padding=(1,0));  # size Y = X + 2P - W + 1 = 7 with padding=1
# To preserve input size (Y=X) for a given W, what padding P should we use?

# Stride
@show y3 = conv4(w, x; padding=(1,0), stride=3);  # size Y = 1 + floor((X+2P-W)/S)

# Mode
@show y4 = conv4(w, x, mode=0);  # Default mode (convolution) inverts w
@show y5 = conv4(w, x, mode=1);  # mode=1 (cross-correlation) does not invert w

# Convolution in more dimensions
x = reshape([1.0:9.0...], (3,3,1,1))

w = reshape([1.0:4.0...], (2,2,1,1))

y = conv4(w, x)

# Convolution with multiple channels, filters, and instances
# size X = [X1,X2,...,Xd,Cx,N] where d is the number of dimensions, Cx is channels, N is instances
x = reshape([1.0:18.0...], (3,3,2,1))

# size W = [W1,W2,...,Wd,Cx,Cy] where d is the number of dimensions, Cx is input channels, Cy is output channels
w = reshape([1.0:24.0...], (2,2,2,3));

# size Y = [Y1,Y2,...,Yd,Cy,N]  where Yi = 1 + floor((Xi+2Pi-Wi)/Si), Cy is channels, N is instances
y = conv4(w,x)

# Pooling operator in Knet
using Knet: pool
@doc pool

# 1-D pooling example
@show x = reshape([1.0:6.0...], (6,1,1,1))
@show pool(x);

# Window size
@show pool(x; window=3);  # size Y = floor(X/W)

# Padding
@show pool(x; padding=(1,0));  # size Y = floor((X+2P)/W)

# Stride
@show x = reshape([1.0:10.0...], (10,1,1,1));
@show pool(x; stride=4);  # size Y = 1 + floor((X+2P-W)/S)

# Mode (using KnetArray here; not all modes are implemented on the CPU)
# using Knet: KnetArray
x = reshape([1.0:6.0...], (6,1,1,1))
@show x
@show pool(x; padding=(1,0), mode=0)  # max pooling
@show pool(x; padding=(1,0), mode=1)  # avg pooling
# @show pool(x; padding=(1,0), mode=2); # avg pooling excluding padded values (is not implemented on CPU)

# More dimensions
x = reshape([1.0:16.0...], (4,4,1,1))

pool(x)

# Multiple channels and instances
x = reshape([1.0:32.0...], (4,4,2,1))

# each channel and each instance is pooled separately
pool(x)  # size Y = (Y1,...,Yd,Cx,N) where Yi are spatial dims, Cx and N are identical to input X

# Load data (see 02.mnist.ipynb)
using Knet: Knet, KnetArray, gpu, minibatch
include(Knet.dir("data","mnist.jl"))  # Load data
dtrn,dtst = mnistdata();              # dtrn and dtst = [ (x1,y1), (x2,y2), ... ] where xi,yi are minibatches of 100

(x,y) = first(dtst)
summary.((x,y))

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
        train!(model, dtrn; callback=callback, optimizer=SGD(lr=0.1), o...)
        Knet.save(file,"results",reshape(results, (4,:)))
    end
    isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
    results = Knet.load(file,"results")
    println(minimum(results,dims=2))
    return results
end

# Redefine Linear layer (See 03.lin.ipynb):
using Knet: param, param0
struct Linear; w; b; end
(f::Linear)(x) = (f.w * mat(x) .+ f.b)
mat(x)=reshape(x,:,size(x)[end])  # Reshapes 4-D tensor to 2-D matrix so we can use matmul
Linear(inputsize::Int,outputsize::Int) = Linear(param(outputsize,inputsize),param0(outputsize))

# Define a convolutional layer:
struct Conv; w; b; end
(f::Conv)(x) = pool(conv4(f.w,x) .+ f.b)
Conv(w1,w2,cx,cy) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1))

# Define a convolutional neural network:
struct CNN; layers; end

# Weight initialization for a multi-layer convolutional neural network
# h[i] is an integer for a fully connected layer, a triple of integers for convolution filters and tensor inputs
# use CNN(x,h1,h2,...,hn,y) for a n hidden layer model
function CNN(h...)
    w = Any[]
    x = h[1]
    for i=2:length(h)
        if isa(h[i],Tuple)
            (x1,x2,cx) = x
            (w1,w2,cy) = h[i]
            push!(w, Conv(w1,w2,cx,cy))
            x = ((x1-w1+1)÷2,(x2-w2+1)÷2,cy) # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
        elseif isa(h[i],Integer)
            push!(w, Linear(prod(x),h[i]))
            x = h[i]
        else
            error("Unknown layer type: $(h[i])")
        end
    end
    CNN(w)
end;

using Knet: dropout, relu
function (m::CNN)(x; pdrop=0)
    for (i,layer) in enumerate(m.layers)
        p = (i <= length(pdrop) ? pdrop[i] : pdrop[end])
        x = dropout(x, p)
        x = layer(x)
        x = (layer == m.layers[end] ? x : relu.(x))
    end
    return x
end

lenet = CNN((28,28,1), (5,5,20), (5,5,50), 500, 10)
summary.(l.w for l in lenet.layers)

using Knet: nll
(x,y) = first(dtst)
nll(lenet,x,y)

using Plots; default(fmt=:png,ls=:auto)
ENV["COLUMNS"] = 92

cnn = trainresults("cnn.jld2", lenet; pdrop=(0,0,.3)); # 406s [8.83583e-5, 0.017289, 0.0, 0.0048]
