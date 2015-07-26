module NeuralNets

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
    decat,
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
    momentum!,
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
    @autograd,
    @paramdef

# gradcheck.jl
export
    gradcheck

include("neuralnet.jl")

# extras.jl in wrapped in a module to 
# prevent namespace pollution
module Extras
    include("extras.jl")
end

include("debug.jl")
include("ops.jl")
include("fit.jl")
include("initialization.jl")
include("loss.jl")
include("grad.jl")
include("gradcheck.jl")

end # module NeuralNets