# Shakespeare
using Pkg
for p in ("Knet","ProgressMeter")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end
RNNTYPE = :lstm
BATCHSIZE = 256
SEQLENGTH = 100
INPUTSIZE = 168
VOCABSIZE = 84
HIDDENSIZE = 334
NUMLAYERS = 1
DROPOUT = 0.0
LR=0.001
BETA_1=0.9
BETA_2=0.999
EPS=1e-08
EPOCHS = 30
ENV["COLUMNS"]=92;
# Load 'The Complete Works of William Shakespeare'
using Knet: Knet
include(Knet.dir("data","gutenberg.jl"))
trn,tst,chars = shakespeare()
map(summary,(trn,tst,chars))
# Print a sample
println(string(chars[trn[1020:1210]]...))
# Minibatch data
using Knet: minibatch
function mb(a)
    N = length(a) ÷ BATCHSIZE
    x = reshape(a[1:N*BATCHSIZE],N,BATCHSIZE)' # reshape full data to (B,N) with contiguous rows
    minibatch(x[:,1:N-1], x[:,2:N], SEQLENGTH) # split into (B,T) blocks
end
dtrn,dtst = mb.((trn,tst))
length.((dtrn,dtst))
summary.(first(dtrn))  # each x and y have dimensions (BATCHSIZE,SEQLENGTH)
using Knet: param, param0, RNN, dropout
struct CharLM; input; rnn; output; end

CharLM(vocab::Int,input::Int,hidden::Int; o...) =
    CharLM(Embed(vocab,input), RNN(input,hidden; o...), Linear(hidden,vocab))

function (c::CharLM)(x; pdrop=0, hidden=nothing)
    x = c.input(x)                # (B,T)->(X,B,T)
    x = dropout(x, pdrop)
    x = c.rnn(x, hidden=hidden)   # (H,B,T)
    x = dropout(x, pdrop)
    x = reshape(x, size(x,1), :)  # (H,B*T)
    return c.output(x)            # (V,B*T)
end
struct Embed; w; end

Embed(vocab::Int,embed::Int)=Embed(param(embed,vocab))

(e::Embed)(x) = e.w[:,x]

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * x .+ l.b

# For running experiments
using Knet: train!, Adam, AutoGrad; import ProgressMeter
function trainresults(file,model,chars)
    if (print("Train from scratch? ");readline()[1]=='y')
        updates = 0; prog = ProgressMeter.Progress(EPOCHS * length(dtrn))
        callback(J)=(ProgressMeter.update!(prog, updates); (updates += 1) <= prog.n)
        opt = Adam(lr=LR, beta1=BETA_1, beta2=BETA_2, eps=EPS)
        Knet.train!(model, dtrn; callback=callback, optimizer=opt, pdrop=DROPOUT, hidden=[])
        Knet.gc(); Knet.save(file,"model",model,"chars",chars)
    else
        isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
        model,chars = Knet.load(file,"model","chars")
    end
    return model,chars
end

clm,chars = trainresults("shakespeare.jld2",
    CharLM(VOCABSIZE, INPUTSIZE, HIDDENSIZE; rnnType=RNNTYPE, numLayers=NUMLAYERS, dropout=DROPOUT), chars);
