# NeuralNets.jl

NeuralNets.jl is a Julia package for describing and training neural networks. NeuralNet.jl aims to allow arbitrary differentiable models with a scalar loss to be expressed and trained without requiring the user to write any model specific backpropagation code. The cost of this flexibility is model specification is completely up to the user.

## Example
Here is an example linear regression model using NeuralNets.jl
```julia
# define model
using NeuralNets
const nnx = NeuralNets.Extras
const n_classes, n_features, n_samples = 3, 20, 100

model = NeuralNet()
model[:w] = Zeros(n_classes, n_features)
model[:b] = Zeros(n_classes)
```

```julia
# define how the model makes predictions
function predict(input, target)
    @paramdef model w b
    x = Block(input)
    @grad model begin
        prediction = affine(w, x, b)
        cost = nll_categorical(target, prediction)
    end
    return nnx.argmax(prediction)
end
```

```julia
# generate some random data and fit model parameters
const X, Y = nnx.gaussblobs(n_classes, n_features, n_samples)
for epoch = 1:100
    Y_pred = predict(X, Y)
    backprop(model)
    sgd!(model, 0.1 / n_samples)
    errors = sum(Y .!= Y_pred)
    println("epoch => $epoch, errors => $errors")
    errors > 0 || break
end
```

This can sometimes get a little messy, especially because NeuralNets.jl employs a functional style notation, i.e., `linear(W, x)` vs `W * x`. The hope is that because nothing is hidden by operator overloading or behind the scenes black magic, it will ultimately be easier to right bug-free and easily extensible code without fighting syntax.

The overall technique is essentially the same stack based approach employed by Andrej Karpathy's [recurrentjs](https://github.com/karpathy/recurrentjs). As computation occurs tracks the application of operators.  and must be supplied as the first argument of every function call. The operator then internally handles how its application changes backpropagation by pushing functions onto a backpropagation stack.


