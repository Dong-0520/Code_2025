using SBPLite
using SafeTestsets, Test

@time begin
    @time @safetestset "sbp.jl tests" include("test_sbp.jl")
    @time @safetestset "curvilinear.jl tests" include("test_curvilinear.jl")
    @time @safetestset "geometry.jl tests" include("test_geometry.jl")
end
