################################################
# sIPOPT test
################################################

using JuMP
using SparseArrays
using LinearAlgebra
using Ipopt
using Random
# using MadNLP
# using KNITRO

include("nlp_utilities.jl")
include("nlp_utilities_test.jl")
include("opf.jl")


test_compute_optimal_hess_jacobian()

test_compute_derivatives()

test_compute_derivatives_Finite_Diff()

test_compute_derivatives_Analytical()

# "pglib_opf_case5_pjm" "pglib_opf_case14_ieee" "pglib_opf_case30_ieee" "pglib_opf_case57_ieee" "pglib_opf_case118_ieee" "pglib_opf_case300_ieee" "pglib_opf_case24_ieee_rts"
Random.seed!(1234)
@time test_bilevel_ac_strategic_bidding("pglib_opf_case24_ieee_rts"; percen_bidding_nodes=0.1)

# Random.seed!(1234)
# Δp=0.0 (no derivative): time=4.61s | obj= $474.18
# Δp=nothing (no restoration): time=452.65s | obj= $79886.16
# Δp=0.001 (with derivative): time=54.64s | obj= $474.18

Δs_primal, Δs_dual = sesitivity_load("pglib_opf_case24_ieee_rts")
Δs_dual[21,1:24]
Δs_dual[21,25:end]