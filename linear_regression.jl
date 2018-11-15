W = rand(2,5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x,y)
    y_t = predict(x)
    sum((y .- y_t).^2)
end

x, y = rand(5), rand(2)
loss(x,y)

using Flux.Tracker

W=param(W)
b=param(b)

gs = Tracker.gradient(()->loss(x,y), Params([W,b]))

using Flux.Tracker: update!

Delta = gs[W]

update!(W,-0.1Delta)

loss(x,y)
