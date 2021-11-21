include("src/ADEngine.jl")
using .ADEngine: Node, backward, relu


a = Node(-4.0)
b = Node(2.0)
c = a + b
d = a * b + b^3
c += c + 1.
c += 1. + c + (-1.0 * a)
d += d * 2. + relu(b + a)
d += 3. * d + relu(b - a)
e = c - d
f = e^2
g = f / 2.0
g += 10.0 / f
println("$(g.data)")  # prints 24.7041, the outcome of this forward pass
backward(g)
println("$(a.grad)") # prints 138.8338, i.e. the numerical value of dg/da
println("$(b.grad)") # prints 645.5773, i.e. the numerical value of dg/db