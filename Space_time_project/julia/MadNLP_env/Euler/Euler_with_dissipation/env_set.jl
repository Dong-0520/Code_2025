using Pkg

# 项目环境路径（按你仓库的层级）
const ENV_PATH = joinpath(ENV["HOME"],
    "Code_2025", "Space_time_project", "julia", "MadNLP_env")
Pkg.activate(ENV_PATH; shared=false)

# 统一注册表（可选）
Pkg.Registry.update()

# 先把之前错误固定的 HSL_jll 清掉（无则忽略）
try
    Pkg.rm(PackageSpec(name="HSL_jll"); mode=Pkg.Types.PKGMODE_MANIFEST)
catch
end

# 优先使用你仓库里随附的 HSL_jll（自带 MA57 等）
const CUSTOM_HSL = joinpath(ENV["HOME"], "Code_2025", "HSL_jll", "HSL_jll.jl.v2024.11.28")
if isdir(CUSTOM_HSL)
    @info "Using custom HSL_jll at $CUSTOM_HSL"
    Pkg.develop(PackageSpec(path=CUSTOM_HSL))
else
    @warn "Custom HSL_jll not found at $CUSTOM_HSL; falling back to registry HSL_jll"
    Pkg.add("HSL_jll")
end

# 分两批安装依赖（去掉 stdlib）
Pkg.add([
    "Trixi", "BlockArrays", "JLD2", "DoubleFloats",
    "ProgressBars", "LoopVectorization", "ArraysOfArrays",
    "SparseConnectivityTracer", "ADTypes", "NonlinearSolve", "NLsolve", "Plots"
])

Pkg.add([
    "ExaModels", "JuMP", "MadNLP", "MadNLPHSL"
])

# 只恢复依赖（不预编译，避免高内存峰值）
Pkg.instantiate()
# 需要时再单独开一次预编译作业：
# Pkg.precompile()
