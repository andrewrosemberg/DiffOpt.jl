################################################
# sIPOPT test
################################################

using JuMP
using SparseArrays
using LinearAlgebra
using Ipopt
# using MadNLP
# using KNITRO

include("nlp_utilities.jl")
include("nlp_utilities_test.jl")


test_compute_optimal_hess_jacobian()

test_compute_derivatives()

test_compute_derivatives_Finite_Diff()