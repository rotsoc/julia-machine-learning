a=rand(10^7)
a'
sum(a)
using Pkg
for p in ("BenchmarkTools","Plots","PyCall","Conda")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end
@time sum(a)
@time sum(a)
@time sum(a)
using BenchmarkTools
using Libdl

C_code = """
#include <stddef.h>
double c_sum(size_t n, double *X) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += X[i];
    }
    return s;
}
"""

const Clib = tempname()   # make a temporary file
# compile to a shared library by piping C_code to gcc
# (works only if you have gcc installed):
open(`gcc -std=c99 -fPIC -O3 -msse3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    print(f, C_code)
end
# define a Julia function that calls the C function:
c_sum(X::Array{Float64}) = ccall(("c_sum", Clib), Float64, (Csize_t, Ptr{Float64}), length(X), X)
c_sum(a)

c_sum(a) ≈ sum(a)

c_sum(a) - sum(a)

c_bench = @benchmark c_sum($a)
println("C: Fastest time was $(minimum(c_bench.times) / 1e6) msec")
d = Dict()  # a "dictionary", i.e. an associative array
d["C"] = minimum(c_bench.times) / 1e6  # in milliseconds
d

using PyCall
apy_list = PyCall.array2py(a, 1, 1)
pysum = pybuiltin("sum")
pysum(a)
pysum(a) ≈ sum(a)
py_list_bench = @benchmark $pysum($apy_list)
d["Python built-in"] = minimum(py_list_bench.times) / 1e6
d

using Conda
numpy_sum = pyimport("numpy")["sum"]
apy_numpy = PyObject(a) # converts to a numpy array by default
numpy_sum(apy_list)
py_numpy_bench = @benchmark $numpy_sum($apy_numpy)

numpy_sum(apy_list) ≈ sum(a)

d["Python numpy"] = minimum(py_numpy_bench.times) / 1e6
d

py"""
def py_sum(A):
    s = 0.0
    for a in A:
        s += a
    return s
"""

sum_py = py"py_sum"

sum_py(apy_list)
py_hand = @benchmark $sum_py($apy_list)
sum_py(apy_list) ≈ sum(a)
d["Python hand-written"] = minimum(py_hand.times) / 1e6
d

@which sum(a)
sum(a)
j_bench = @benchmark sum($a)
d["Julia built-in"] = minimum(j_bench.times) / 1e6
d

function mysum(A)
    s = 0.0 # s = zero(eltype(a))
    for a in A
        s += a
    end
    s
end
mysum(a)
j_bench_hand = @benchmark mysum($a)
d["Julia hand-written"] = minimum(j_bench_hand.times) / 1e6
d

function mysum_simd(A)
    s = 0.0 # s = zero(eltype(A))
    @simd for a in A
        s += a
    end
    s
end

mysum_simd(a)
j_bench_hand_simd = @benchmark mysum_simd($a)


d["Julia hand-written simd"] = minimum(j_bench_hand_simd.times) / 1e6
d

for (key, value) in sort(collect(d), by=last)
    println(rpad(key, 25, "."), lpad(round(value, digits=1), 6, "."))
end
