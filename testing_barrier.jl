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

# No Fix

test_compute_optimal_hess_jacobian()

test_compute_derivatives_Finite_Diff(DICT_PROBLEMS_no_cc, false)

test_compute_derivatives_Analytical(DICT_PROBLEMS_Analytical_no_cc)

# Fix and Relax

test_compute_derivatives()

test_compute_derivatives_Finite_Diff(DICT_PROBLEMS_cc, true)

test_compute_derivatives_Analytical(DICT_PROBLEMS_Analytical_cc)

################################################
# Strategic bidding test
################################################

# "pglib_opf_case5_pjm" "pglib_opf_case14_ieee" "pglib_opf_case30_ieee" "pglib_opf_case57_ieee" "pglib_opf_case118_ieee" "pglib_opf_case300_ieee" "pglib_opf_case24_ieee_rts"
# Random.seed!(1234)
# @time test_bilevel_ac_strategic_bidding("pglib_opf_case24_ieee_rts"; percen_bidding_nodes=0.1)

casename = "pglib_opf_case300_ieee"
Random.seed!(1234)
@time test_bilevel_ac_strategic_bidding(casename; percen_bidding_nodes=0.1, Δp=nothing)
# Δp=0.0 (no derivative): time=4.61s | obj= $474.18
# Δp=nothing (no restoration): time=452.65s | obj= $79886.16
# Δp=0.001 (with derivative): time=54.64s | obj= $474.18

# casename = "pglib_opf_case14_ieee"
# Random.seed!(1234)
# @time test_bilevel_ac_strategic_bidding(casename; percen_bidding_nodes=0.1, Δp=nothing)
# Δp=0.0 (no derivative): time=8.93s | obj= $0.00
# Δp=nothing (no restoration): time=0.94s | obj= $0.00
# Δp=0.001 (with derivative): time=0.58s | obj= $0.00



################################################
# Load sensitivity
################################################

Δs_primal, Δs_dual = sesitivity_load("pglib_opf_case24_ieee_rts")
Δs_dual[21,1:24]
Δs_dual[21,25:end]

################################################
# Moonlander test
################################################

include("moonlanding.jl")
using HSL_jll

Isp_0 = 0.3
model, tf, Isp = moonlander_JMP(_I=Isp_0)
set_optimizer(model, optimizer_with_attributes(Ipopt.Optimizer, 
    "print_level" => 0,
    "hsllib" => HSL_jll.libhsl_path,
    "linear_solver" => "MA57"
))
JuMP.optimize!(model)
termination_status = JuMP.termination_status(model)
tf_0 = value(tf)

function compute_sentitivity_for_var(model; _primal_vars=[], _cons=[], Δp=nothing)
    primal_vars = all_primal_vars(model)
    params = all_params(model)
    vars_idx = [findall(x -> x == i, primal_vars)[1] for i in _primal_vars]
    evaluator, cons = create_evaluator(model; x=[primal_vars; params])
    num_cons = length(cons)
    cons_idx = [findall(x -> x == i, cons)[1] for i in _cons]
    leq_locations, geq_locations = find_inequealities(cons)
    num_ineq = length(leq_locations) + length(geq_locations)
    num_primal = length(primal_vars)

    Δs, sp = compute_sensitivity(evaluator, cons, Δp; primal_vars=primal_vars, params=params)
    return Δs[vars_idx, :], Δs[num_primal+num_ineq.+cons_idx, :], sp[vars_idx, :], sp[num_primal+num_ineq.+cons_idx, :]
end

Δp=nothing

Δs_primal, Δs_dual, sp_primal, sp_dual = compute_sentitivity_for_var(model; _primal_vars=[tf], Δp=Δp)
Δp = 0.001
tf_p = tf_0 + Δs_primal[1] * Δp

set_parameter_value(Isp, Isp_0+Δp)
JuMP.optimize!(model)
termination_status = JuMP.termination_status(model)
tf_p_true = value(tf)

@test tf_p ≈ tf_p_true rtol=1e-3