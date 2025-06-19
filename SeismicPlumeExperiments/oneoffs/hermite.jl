
using CairoMakie

h0(x) = exp(-x^2/2)
h1(x) = - x * exp(-x^2/2)
h2(x) = 1 / sqrt(factorial(2)) * (1 - x^2) * exp(-x^2/2)
h3(x) = 1 / sqrt(factorial(3)) * (3*x - x^3) * exp(-x^2/2)
h4(x) = 1 / sqrt(factorial(4)) * (x^4 - 6*x^2 + 3) * exp(-x^2/2)
h9(x) = 1 / sqrt(factorial(9)) * (x^9 - 36*x^7 + 378 * x^5 - 1260*x^3 + 945*x) * exp(-x^2/2)
h10(x) = 1 / sqrt(factorial(10)) * (x^10 - 45*x^8 + 630*x^6 - 3150 * x^4 + 4725*x^2 - 945) * exp(-x^2/2)

xs = -6:0.05:6

fig = Figure();
ax = Axis(fig[1, 1]);

lines!(ax, xs, h0.(xs), label="h0")
lines!(ax, xs, h1.(xs), label="h1")
lines!(ax, xs, h2.(xs), label="h2")
lines!(ax, xs, h3.(xs), label="h3")
lines!(ax, xs, h4.(xs), label="h4")
lines!(ax, xs, h9.(xs), label="h9")
lines!(ax, xs, h10.(xs), label="h10")

Legend(fig[1, 2], ax)
display(fig)
