tests = [
    "test_utils",
    "test_ffann"
]

srand(sum(map(int, collect("NeuralNet"))))

global failed = false

test_handler(r::Test.Success) = nothing

function test_handler(r::Test.Failure)
    global failed
    if !failed
        println()
        failed = true
    end
    println("fail: $(r.expr)")
end

function test_handler(r::Test.Error)
    global failed
    if !failed
        println()
        failed = true
    end
    println("fail: $(showerror(STDOUT, r))")
end

for t in tests
    global failed
    failed = false
    f = "$t.jl"
    print_with_color(:green, "* $t...")
    Test.with_handler(test_handler) do
        include(f)
    end
    if !failed
        print_with_color(:green, " ok\n")
    else
        print_with_color(:red, "failure\n")
    end

end