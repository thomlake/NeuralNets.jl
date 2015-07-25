# NeuralNets.jl

NeuralNets.jl is a Julia package for describing and training neural networks. NeuralNet.jl aims to allow arbitrary differentiable models with a scalar loss to be expressed and trained without requiring the user to write any model specific backpropagation code. The cost of this flexibility is model specification is completely up to the user.

## Example

```julia
# example linear regression model in NueralNets.jl
using NeuralNets
model = NeuralNet([:w => Zeros(n_outputs, n_features), :b => Zeros(n_outputs)])
function predict(input, target)
    @grad model begin
        prediction = affine(model[:w], Block(input), model[:b])
        cost = nll_normal(target, prediction)
    end
end
for i = 1:n_iters
    predict(X, Y)
    backprop(model)
    sgd!(model, 0.01 / size(X, 2))
end
```

This can sometimes get a little messy, especially because NeuralNets.jl employs a functional style notation, i.e., `linear(W, x)` vs `W * x`. The hope is that because nothing is hidden by operator overloading or behind the scenes black magic, it will ultimately be easier to right bug-free and easily extensible code without fighting syntax.

The overall technique is essentially the same stack based approach employed by Andrej Karpathy's [recurrentjs](https://github.com/karpathy/recurrentjs). As computation occurs tracks the application of operators.  and must be supplied as the first argument of every function call. The operator then internally handles how its application changes backpropagation by pushing functions onto a backpropagation stack.


