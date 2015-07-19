# Linear Regression example using NeuralNets.jl package
# author: tllake 
# email: thom.l.lake@gmail.com
#
# This example is for demonstration purposes only.
# Using gradient descent to fit a linear regression
# model on a dataset like Boston Housing is a poor choice.

using RDatasets
using MLTools
using NeuralNets

function Model(n_in::Int, n_out::Int)
    # A NeuralNet instance needs 3 parts
    # - A Graph instance to handle backpropagation.
    # - Parameters to be optimized.
    # - A feedforward function that defines the computation 
    #   performed by the model.

    G = Graph()
    W = Param(n_out, n_in)
    b = Param(n_out)
    
    # Define 2 versions of feedforward. 
    # The first version of feedforward is for training. 
    # It accepts an input and output and applies a loss function.
    # The second version is for making predictions. 
    # It accepts an input and returns the prediction.
    function feedforward(X::Matrix, Y_true::Matrix)
        X = Block(X)
        Y_pred = add(G, linear(G, W, X), b)
        mseloss(Y_true, Y_pred)
    end

    function feedforward(X::Matrix)
        X = Block(X)
        Y_pred = add(G, linear(G, W, X), b)
        value(Y_pred)
    end

    NeuralNet(G, [:W => W, :b => b], feedforward)
end

function test()
    # Example of how to apply finite difference
    # gradient checking to a NeuralNet instance.
    X = randn(10, 2)
    Y = randn(3, 2)
    model = Model(size(X, 1), 3)
    fwd() = model.feedforward(X, Y)
    gradcheck(model, fwd)
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

    model = Model(n_features, n_outputs)
    loss_prev = Inf
    converged = false
    epochs = 0

    # Fit parameters
    while !converged
        loss = model.feedforward(X, Y)[1]
        backprop(model)
        sgd(model, 0.01 / n_samples)
        if loss >= loss_prev
            converged = true
        end
        loss_prev = loss
        epochs += 1
    end

    println("[Info]")
    println("  number of samples: $n_samples")
    println("  number of features: $n_features")
    println("  converged after $epochs updates")
    println("  avg loss: $(loss_prev / n_samples)")
    println("[Coefficients]")
    for (k, v) in zip(expl, value(getparam(model, :W)))
        println("  $(rpad(k, 8)) => $(sign(v) > 0 ? "+" : "-")$(round(abs(v), 3))")
    end
end

demo()
