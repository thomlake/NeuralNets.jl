module NeuralNets

# Utilities
export 
    vec2mat,
    onehot,
    backprop

# Types
export 
    Graph,
    NeuralNet,
    Block,
    getparam,
    value

# Graph Ops
export 
    tanh,
    sigmoid,
    relu,
    softmax,
    maxout,
    concat,
    linear,
    mult,
    add,
    minus,
    dropout

# Loss Functions
export
    mseloss,
    catloss

# Fitting
export
    sgd,
    rmsprop

# Initialization
export 
    register_param_initializer!,
    Param

export
    gradcheck

include("graph.jl")
include("fit.jl")
include("initialization.jl")
include("utils.jl")


end # module NeuralNets