# Linear Regression example using NeuralNets.jl package
# author: tllake 
# email: thom.l.lake@gmail.com
#
# This example is for demonstration purposes only.
# Using gradient descent to fit a linear regression
# model on a dataset like Boston Housing is a poor choice.

using RDatasets
using SoItGoes
using NeuralNets

function build_model(n_features, n_outputs)
    model = NeuralNet(Symbol)
    model[:w] = Zeros(n_outputs, n_features)
    model[:b] = Zeros(n_outputs)
    return model
end

function predict(model::NeuralNet, input::Matrix, target::Matrix)
    w = model[:w]
    b = model[:b]
    x = Block(input)
    @grad model begin
        prediction = affine(w, x, b)
        cost = nll_normal(target, prediction)
    end
    return cost
end

predict(model::NeuralNet, input::Matrix) = affine(model[:w], Block(input), model[:b])

function test()
    # Example of how to apply finite difference
    # gradient checking to a NeuralNet instance.
    n_samples = 2
    n_features = 10
    n_outputs = 3
    X = randn(n_features, n_samples)
    Y = randn(n_outputs, n_samples)
    model = build_model(n_features, n_outputs)
    gradcheck(model, ()->predict(model, X, Y))
end

function demo()
    srand(123)

    # Load some data.
    df = dataset("MASS", "boston")
    resp = [:MedV]
    expl = [:Rm, :Crim, :LStat, :PTRatio, :Dis]
    
    # Convert data frames to Arrays and normalize to have
    # zero mean and unit stard deviation. Notice the arrays
    # are transposed so columns are instances and rows are features.
    X = normalize(convert(Array{Float64}, df[:, expl]).', 2)
    Y = normalize(convert(Array{Float64}, df[:, resp]).', 2)

    n_samples = size(X, 2)
    n_features = length(expl)
    n_outputs = length(resp)
    model = build_model(n_features, n_outputs)

    cost_prev = Inf
    converged = false
    epochs = 0

    # Fit parameters
    t0 = time()
    while !converged
        cost = predict(model, X, Y)
        backprop(model)
        sgd!(model, 0.01 / n_samples)
        if cost >= cost_prev
            converged = true
        end
        cost_prev = cost
        epochs += 1
    end
    t1 = time()

    println("[Info]")
    println("  number of samples: $n_samples")
    println("  number of features: $n_features")
    println("  converged after $epochs updates ($(round(t1 - t0, 2)) seconds)")
    println("  avg loss: $(cost_prev / n_samples)")
    println("[Coefficients]")
    for (k, v) in zip(expl, value(model[:w]))
        println("  $(rpad(k, 8)) => $(sign(v) > 0 ? "+" : "-")$(round(abs(v), 3))")
    end
end

test()
demo()
