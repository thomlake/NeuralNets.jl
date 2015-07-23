module NeuralNets

# utils.jl
export 
    vec2mat,
    onehot,
    gradcheck

# neuralnet.jl
export 
    NeuralNet,
    Block,
    value,
    backprop

# ops.jl
export
    dropout,
    tanh,
    sigmoid,
    relu,
    softmax,
    maxout,
    concat,
    linear,
    mult,
    add,
    minus

# loss.jl
export
    nll_normal,
    nll_categorical

# fit.jl
export
    sgd!,
    rmsprop!

# initialization.jl
export 
    Normal,
    Uniform,
    Glorot,
    Orthonormal,
    Sparse,
    Identity,
    Zeros

export
    @grad

include("debug.jl")
include("neuralnet.jl")
include("utils.jl")
include("ops.jl")
include("fit.jl")
include("initialization.jl")
include("loss.jl")
include("grad.jl")


end # module NeuralNets