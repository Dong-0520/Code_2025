using Pkg
using Roots       # for find_zero, bisection
using Plots       # for plotting

function solve_u(x, t; u_init::Float64 = 0.0)
    # Define F(u) = u - sin(4πx - 4πu*t)
    F(u) = u - sin(4π*x - 4π*u*t)
    
    # We expect a solution to be in [-1, 1].
    # We'll do a simple bisection or bracketed approach:
    return find_zero(F, (-1.0, 1.0), Bisection(), atol=1e-15)
end


# Create a range of x and t values
xs = range(0, 0.5, length=50)
ts = range(0, 0.1, length=50)

# Pre-allocate a matrix to store the solution u(t, x)
# We often store as U[t-index, x-index] in Julia
U = [ solve_u(x, t) for t in ts, x in xs ]


# Make a surface plot of u, as a function of (x,t).
surface(
    xs, ts, U, 
    xlabel="x",
    ylabel="t",
    zlabel="u",
    title="Solution of u = sin(4πx - 4πu·t)"
)

