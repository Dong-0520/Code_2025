
using Pkg
Pkg.activate("MadNLP_env")


#---# MadNLP 相关
using MadNLP
using NLPModels
struct HS15Model <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    params::Vector{Float64}   
end


function HS15Model(x0, params = [100.0, 1.0])
    return HS15Model(
        NLPModels.NLPModelMeta(
            2,
            ncon = 2,
            nnzj = 4,
            nnzh = 3,
            x0 = x0,
            y0 = zeros(2),
            lvar = [-Inf, -Inf],
            uvar = [0.5, Inf],
            lcon = [1.0, 0.0],
            ucon = [Inf, Inf],
            minimize = true
        ),
        NLPModels.Counters(),
        params    # ⬅️ 存入 params
    )
end

function NLPModels.obj(nlp::HS15Model, x::AbstractVector)
    p1, p2 = nlp.params  # ⬅️ 拿出参数
    return p1 * (x[2] - x[1]^2)^2 + (p2 - x[1])^2
end


function NLPModels.grad!(nlp::HS15Model, x::AbstractVector, g::AbstractVector)
    p1, p2 = nlp.params
    z = x[2] - x[1]^2
    g[1] = -2 * p1 * z * 2 * x[1] - 2 * (p2 - x[1])
    g[2] = 2 * p1 * z
    return g
end


function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
end

function NLPModels.jac_coord!(nlp::HS15Model, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]    # (1, 1)
    J[2] = x[1]    # (1, 2)
    J[3] = 1.0     # (2, 1)
    J[4] = 2*x[2]  # (2, 2)
    return J
end

function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
end

function NLPModels.hess_coord!(
    nlp::HS15Model,
    x, y,
    H::AbstractVector{Float64};
    obj_weight::Float64 = 1.0
)
    p1, p2 = nlp.params

    # 目标函数 H 展开
    H[1] = obj_weight * ( -4*p1*x[2] + 12*p1*x[1]^2 + 2.0 )
    H[2] = obj_weight * ( -4*p1*x[1] )
    H[3] = obj_weight * (  2*p1  )

    # 约束1 c₁=x₁*x₂ ，添加到 H[2]
    H[2] += y[1] * 1.0 

    # 约束2 c₂=x₁ + x₂^2 ，添加到 H[3]
    H[3] += y[2] * 2.0

    return H
end


x0 = [0.0, 0.0]
params1 = [100.0, 1.0]
nlp = HS15Model(x0, params1)

using MadNLP
solver = MadNLP.MadNLPSolver(nlp)
results = MadNLP.solve!(solver)


println("Objective: ", results.objective)
println("Solution:  ", results.solution)




# 👇 改参数、warm-start第二次求解
nlp.params .= [110.0, 1.0]    # 小幅改动
solver2 = MadNLP.MadNLPSolver(nlp; x0 = results.solution, y0 = results.multipliers)
results2 = MadNLP.solve!(solver2)


results2.solution


nlp.params .= [101.0, 1.1]
copyto!(NLPModels.get_x0(nlp), results.solution)


#-----------------------------------------------------------
# https://madnlp.github.io/MadNLP.jl/dev/tutorials/lbfgs/
using MadNLP, ExaModels, Random
function elec_model(np)
    Random.seed!(1)
    # Set the starting point to a quasi-uniform distribution of electrons on a unit sphere
    theta = (2pi) .* rand(np)
    phi = pi .* rand(np)

    core = ExaModels.ExaCore(Float64)
    x = ExaModels.variable(core, 1:np; start = [cos(theta[i])*sin(phi[i]) for i=1:np])
    y = ExaModels.variable(core, 1:np; start = [sin(theta[i])*sin(phi[i]) for i=1:np])
    z = ExaModels.variable(core, 1:np; start = [cos(phi[i]) for i=1:np])
    # Coulomb potential
    itr = [(i,j) for i in 1:np-1 for j in i+1:np]
    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in itr)
    # Unit-ball
    ExaModels.constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i=1:np)

    return ExaModels.ExaModel(core)
end
nh = 10
nlp = elec_model(nh)
results_hess = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
)

results_qn = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
    hessian_approximation=MadNLP.CompactLBFGS,
)



using LinearAlgebra, OpenBLAS32_jll
LinearAlgebra.BLAS.lbt_forward(libopenblas)
using HSL_jll

function LIBHSL_isfunctional()
    @ccall libhsl.LIBHSL_isfunctional()::Bool
end

println(LIBHSL_isfunctional())


#-----------------------------------------------------------
# ExaModel 建模/（变量、目标、约束）
# MadNLP 优化器（自动提取 Jacobian、Hessian）
# [HSL_jll] → 线性求解器（高速稀疏解 KKT 系统）

# 切换 BLAS 后端
# 你必须切换到 LP64（32-bit index）BLAS：
using LinearAlgebra, OpenBLAS32_jll
LinearAlgebra.BLAS.lbt_forward(libopenblas)

using ExaModels, MadNLP, MadNLPHSL, Random
# 以电子模型为例，创建一个 ExaModel
# 该模型包含 np 个电子，目标是最小化它们之间的库仑势能，同时约束它们在单位球内。
# ExaModel 是一个高性能的数学模型框架，适用于大规模优化
function electron_model1(np)
    Random.seed!(1)
    θ = 2π .* rand(np)
    ϕ = π .* rand(np)

    core = ExaModels.ExaCore(Float64)

    x = ExaModels.variable(core, 1:np; start = [cos(θ[i]) * sin(ϕ[i]) for i in 1:np])
    y = ExaModels.variable(core, 1:np; start = [sin(θ[i]) * sin(ϕ[i]) for i in 1:np])
    z = ExaModels.variable(core, 1:np; start = [cos(ϕ[i]) for i in 1:np])

    # Coulomb potential
    itr = [(i,j) for i in 1:np-1 for j in i+1:np]
    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in itr)
    # Unit-ball
    ExaModels.constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i in 1:np)


    return ExaModels.ExaModel(core)
end


function electron_model2(np)
    Random.seed!(1)
    θ = 2π .* rand(np)
    ϕ = π .* rand(np)

    core = ExaModels.ExaCore(Float64)
    x = ExaModels.variable(core, 1:np; start = [cos(θ[i]) * sin(ϕ[i]) for i in 1:np])
    y = ExaModels.variable(core, 1:np; start = [sin(θ[i]) * sin(ϕ[i]) for i in 1:np])
    z = ExaModels.variable(core, 1:np; start = [cos(ϕ[i]) for i in 1:np])


    pairs = [(i, j) for i in 1:np-1 for j in i+1:np]

    # 分为两组：偶数和奇数
    pairs_normal = [(i, j) for (i, j) in pairs if (i + j) % 2 == 0]
    pairs_special = [(i, j) for (i, j) in pairs if (i + j) % 2 != 0]

    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in pairs_normal)
    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in pairs_special)

    ExaModels.constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i in 1:np)

    return ExaModels.ExaModel(core)
end
np = 10  # 先用小规模测试
nlp1 = electron_model1(np)
solver1 = MadNLP.MadNLPSolver(nlp1; hessian = :LDFP)  # 使用 L-BFGS
results1 = MadNLP.solve!(solver1)

nlp2 = electron_model2(np)
solver2 = MadNLP.MadNLPSolver(nlp2; hessian = :LDFP)  # 使用 L-BFGS
results2 = MadNLP.solve!(solver2)

println("优化完成，目标值：", results1.objective)
println("优化完成，目标值：", results2.objective)


#-----------------------------
using JLD2

@load joinpath(@__DIR__, "Euler_small_grid.jld2") grid