
function gradcheck(nnet::NeuralNet, f::Function; eps::FloatingPoint=1e-6, tol::FloatingPoint=1e-6, verbose::Bool=true)
    f()
    backprop(nnet)
    nnet.G.dobackprop = false
    for (name, theta) in nnet.params
        for i = 1:size(theta, 1)
            for j = 1:size(theta, 2)
                xij = theta.x[i,j]
                theta.x[i,j] = xij + eps
                lp = f()
                theta.x[i,j] = xij - eps
                lm = f()
                theta.x[i,j] = xij
                dxij = (lp - lm) / (2 * eps)
                if abs(dxij - theta.dx[i,j]) > tol
                    errmsg = "Finite difference gradient check failed! (name: $name)"
                    errdsc = "|$(dxij) - $(theta.dx[i,j])| > $tol"
                    error("$errmsg\n  $errdsc")
                end
            end
        end
    end
    if verbose
        println("gradcheck passed")
    end
end