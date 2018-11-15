# (c) Deniz Yuret, 2018.
# Note that this is an instructional example written in low-level Julia/Knet and it is slow to train.
# For a faster and high-level implementation please see `@doc RNN`.
# TODO: check the 50% speed regression in julia 1.0.
using Pkg; haskey(Pkg.installed(),"Knet") || Pkg.add("Knet")
using Knet; #@show Knet.gpu()

# Comparison of a single hidden layer MLP and corresponding RNN

function mlp1(param, input)
    hidden = tanh(input * param[1] .+ param[2])
    output = hidden * param[3] .+ param[4]
    return output
end

function rnn1(param, input, hidden)
    input2 = hcat(input, hidden)
    hidden = tanh(input2 * param[1] .+ param[2])
    output = hidden * param[3] .+ param[4]
    return (hidden, output)
end;
