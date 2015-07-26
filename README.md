# NeuralNets.jl
NeuralNets.jl is a Julia package for describing and training neural networks. NeuralNet.jl aims to allow arbitrary differentiable models with a scalar loss to be expressed natively and trained without requiring the user to write any model specific backpropagation code. The cost of this flexibility is loss of performance compared to other general purpose neural network tools, and leaving the details of model specification completely up to the user.

Following this line of reasoning NeuralNets.jl has two overarching goals. The first is that by making it _relatively_ easy to express complicated models, NeuralNets.jl will allow the space of interesting models (beyond the more common feedforward and slight elaborations on Elman style recurrent neural networks) to be explored more efficiently. For example, models with [attentional components](http://arxiv.org/abs/1409.0473) and more exotic things like [Memory Networks](http://arxiv.org/abs/1503.08895) can be easily expressed in NeuralNets.jl.

The second goal is clarity. The hope is that because nothing is hidden by operator overloading and behind the scenes black magic, the codebase itself will be easier to extend, and programs using NeuralNets.jl will ultimately be easier to debug and write without fighting syntax.

Although similar in spirit, NeuralNets.jl is not intended as a replacement for tools like [Theano](http://deeplearning.net/software/theano/). Some notable differences are:

- No GPU support.
- No operator overloading.
- No automatic computation graph optimization.
- No compilation process (outside of Julia's own JIT).
- Native control flow with `for`, `while`, etc.
- Recursion.
- Dynamic.
- No fundamental difference between parameters and values.
- Functional notation.

## Installation
NeuralNets.jl is currently unregistered, to install use
```julia
julia> Pkg.clone("https://github.com/thomlake/NeuralNets.jl.git")
```

## Example
Let's define a logistic regression model using NeuralNets.jl.
```julia
using NeuralNets
const n_classes, n_features, n_samples = 3, 20, 100
model = NeuralNet()
model[:w] = Zeros(n_classes, n_features)
model[:b] = Zeros(n_classes)
```
We begin by creating an empty `NeuralNet` and then defining parameters. Parameter names can be anything that can be a key in a Dict. The only parameter types currently supported are 2d Arrays. The single argument version of `Zeros` above results in a parameter with size `(n_classes, 1)`.

```julia
const nnx = NeuralNets.Extras
function predict(model, input::Matrix)
    w = model[:w]
    b = model[:b]
    x = Block(input)
    prediction = affine(w, x, b)
    return nnx.argmax(prediction)
end
```
Next we define the computation our model performs when mapping inputs to outputs. Notice the `x = Block(input)` line. This is necessary to allow NeuralNets.jl to incorporate the variable into the computation.

```julia
function predict(model, input::Matrix, target::Vector{Int})
    @paramdef model w b
    x = Block(input)
    @autograd model begin
        prediction = affine(w, x, b)
        cost = nll_categorical(target, prediction)
    end
    return nnx.argmax(prediction)
end
```
The above function defines another version of predict which takes an extra argument, `target`. This function will be used to adjust the parameters of the model to minimize the cost. Having to define two versions of predict may seem verbose, but it is necessary to accommodate cases when the computation for training and testing differ (like when using dropout). There are a few concepts that need explaining here. 

The first is the use of the [`@paramdef`](#the-paramdef-macro) macro. This is just syntactic sugar for defining variables in the current scope. In the above case it is equivalent to writing `w = model[:w]; b = model[:b];`. 

The second is the `@autograd` macro. This tells NeuralNets.jl to collect information in the forward pass required to backpropagate through known operators (see [Operators](#Operators) below for a list) in the given block of code. 

Next we apply a cost function, in this case, the negative log likelihood of a categorical variable. Notice we didn't have to transform `prediction` first by exponentiating and normalizing, i.e. applying a softmax. For computational efficiency NeuralNets.jl internally handles this procedure by applying the correct transformation, similarly to how it would be handled in a generalized linear model (GLM) package.

```julia
const X, Y = nnx.randblobs(n_classes, n_features, n_samples)
for epoch = 1:100
    Y_pred = predict(model, X, Y)
    backprop(model)
    sgd!(model, 0.1 / n_samples)
    errors = sum(Y .!= Y_pred)
    println("epoch => $epoch, errors => $errors")
    errors > 0 || break
end
```
Lastly we write code to generate some artificial data from three `n_features` dimensional diagonal Gaussian distributions with different means and standard deviations, and update model parameters. The three primary components inside the loop above are

- `predict:` map inputs to outputs.
- `backprop:` compute gradients of the cost with respect to the parameters.
- `sgd!:` take a gradient descent step to reduce the value of the cost function.

## Operators
NeuralNets.jl knows how to backpropagate through the following functions:
- `tanh:` hyperbolic tangent function.
- `sigmoid:` logistic sigmoid function.
- `relu:` rectified linear function.
- `softmax:` softmax function.
- `wta:` [winner takes all](http://people.idsia.ch/~juergen/nips2013.pdf)
- `mult:` element-wise multiplication
- `linear:` linear transformation, `W * x`.
- `add:` element-wise addition.
- `minus:` element-wise subtraction.
- `concat:` vector concatenation
- `affine:` affine transformation, `W * x + b`.

## Extensibility
Extending NeuralNets.jl by adding new operators is relatively straightforward. This is especially true when the operator is a convenience wrapper around existing functionality. For example the definition of `affine` in `ops.jl` is simply
```julia
affine(w::Block, x::Block, b::Block) = add(linear(w, x), b)
affine(nnet::NeuralNet, w::Block, x::Block, b::Block) = add(nnet, linear(nnet, w, x), b)
```
The first version handles the case when the call is not wrapped in a `@autograd` and the second handles the case when it is. Notice in the second case the `nnet` argument is passed to each function call. This is necessary to ensure backpropagation works correctly.

When the operator is not composed exclusively of existing functions, the user must also define how to compute gradients. The definition of `tanh` serves well for demonstration purposes.
```julia
Base.tanh(inblock::Block) = Block(tanh(inblock.x))
function bwd_tanh(outblock::Block, inblock::Block)
    inblock.dx += (1 .- (outblock.x .* outblock.x)) .* outblock.dx
end
function Base.tanh(nnet::NeuralNet, inblock::Block)
    outblock = tanh(inblock)
    push!(nnet.bpstack, () -> bwd_tanh(outblock, inblock))
    outblock
end
```

The only remaining thing to do is to let `@autograd` know about the existence of the new operator. To do this simply add the new op to the `nnet_ops` array defined in `grad.jl`.

## The `@paramdef` macro
As noted above, `@paramdef` is syntactic sugar for defining variables in the current scope. It works with parameters whose keys are either symbols, `:theta`, or tuples of symbols and integers, `(:theta, 1, 2)`. In the later case the first element must be a symbol. It will create a variable with tuple elements separated by `_`, i.e. `theta_1_2`. 

The tuple version is generally less useful. The typical use case of parameter keys with integers is programmatic key generation. In this case `@paramdef` maps these programmatically generated keys to back to variable names, which then have to be manipulated by the programmer.

For example consider the following _deep_ neural network.
```julia
const sizes = [n_features, 200, 100, 200, n_outputs]
nnet = NeuralNet()
nnet.metadata[:depth] = length(sizes) - 1
for i = 1:nnet.metadata[:depth]
    nnet[(:w, i)] = Orthonormal(sqrt(2), sizes[i + 1], sizes[i])
    nnet[(:b, i)] = Zeros(sizes[i + 1])
end
```
Using `@paramdef` in the `predict` function would require the programmer to manipulate names like `w_1` and  `w_2`. It is much simpler to just loop through these variables.
```julia
function predict(nnet, input)
    h = Block(input)
    for i = 1:nnet.metadata[:depth]
        w, b = nnet[(:w, i)], nnet[(:b, i)]
        h = relu(affine(w, h, b))
    end
    return nnx.argmax(h)
end
```

## Backpropagation Technique
The overall technique for automating backpropagation is essentially the same stack based approach employed by Andrej Karpathy's [recurrentjs](https://github.com/karpathy/recurrentjs). As computation occurs, NeuralNets.jl tracks the application of known operators. The operator then internally handles how its application changes backpropagation by pushing functions onto a backpropagation stack.


