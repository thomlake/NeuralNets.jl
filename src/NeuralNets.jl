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
    Block

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

include("graph.jl")
include("fit.jl")
include("initialization.jl")


end # module NeuralNets