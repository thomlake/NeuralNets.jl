# NeuralNets.jl

NeuralNets.jl is a Julia package for training neural networks. NeuralNet.jl aims to allow arbitrary differentiable models with a scalar loss to be expressed and trained without requiring the user to write any backpropagation code. 

The cost of this abstraction is model specification is completely up to the user. This can sometimes get a little messy, especially because NeuralNets.jl employs a functional style notation, i.e., `linear(G, W, x)` vs `W * x`. The hope is that because nothing is hidden by operator overloading or behind the scenes black magic, it will ultimately be easier to right bug-free and easily extensible code without fighting syntax.

The overall technique is essentially the same stack based approach employed by Andrej Karpathy's [recurrentjs](https://github.com/karpathy/recurrentjs). A graph `G` tracks the application of operators and must be supplied as the first argument of every function call. The function then internally handles how its application changes backpropagation by pushing functions onto a backpropagation stack.


