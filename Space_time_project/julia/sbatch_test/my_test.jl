#!/usr/bin/env julia

# 打印 Julia 环境信息
println("=== Julia Version & System Info ===")
println(versioninfo())

# 简单数值计算：矩阵乘法
println("=== Running matrix multiply benchmark ===")
using LinearAlgebra, Random

Random.seed!(1234)
A = rand(1000, 1000)
B = rand(1000, 1000)
C = A * B
println("Matrix multiply finished. Size of result: ", size(C))
println("Checksum: ", sum(C))

# 保存到文件（结果写到 mytest_result.txt）
open("mytest_result.txt", "w") do io
    write(io, "Matrix multiply result checksum: $(sum(C))\n")
end

println("✅ mytest.jl finished successfully. Result saved to mytest_result.txt")
