using Base.Test
using NeuralNets

# vec2mat tests
v = randn(20)
m = vec2mat(v)
@test size(m) == (20, 1)
@test all(v .== m)
@test_throws(MethodError, vec2mat(randn(3, 4)))
@test_throws(MethodError, vec2mat(1))

# test onehot
d = 20
for i = 1:d
    v = onehot(i, d)
    for j = 1:d
        @test i == j ? v[j] == 1 : v[j] == 0
    end
end

# test argmax
v = randn(20)
vmax, imax = -Inf, 0
for i = 1:length(v)
    if v[i] > vmax
        vmax = v[i]
        imax = i
    end
end
@test imax == argmax(v)

m = randn(20, 30)
imax = argmax(m)
@test size(imax) == (30,)
for j = 1:size(m, 2)
    @test argmax(m[:,j]) == imax[j]
end

@test_throws(MethodError, argmax(randn(3, 4, 5)))
@test_throws(MethodError, argmax(randn()))

