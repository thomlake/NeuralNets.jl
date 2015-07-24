module NeuralNets

# utils.jl
export 
    vec2mat,
    onehot,
    argmax

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
    affine,
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

# grad.jl
export
    @grad

# gradcheck.jl
export
    gradcheck

include("debug.jl")
include("neuralnet.jl")
include("utils.jl")
include("ops.jl")
include("fit.jl")
include("initialization.jl")
include("loss.jl")
include("grad.jl")
include("gradcheck.jl")


end # module NeuralNets