using JuMP
using Ipopt
using Test
using FiniteDiff

#=
# Test JuMP Hessian and Jacobian

From JuMP Tutorial for Querying Hessians:
https://github.com/jump-dev/JuMP.jl/blob/301d46e81cb66c74c6e22cd89fb89ced740f157b/docs/src/tutorials/nonlinear/querying_hessians.jl#L67-L72
=#
function create_nonlinear_jump_model()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, p ∈ MOI.Parameter(1.0))
    @variable(model, p2 ∈ MOI.Parameter(2.0))
    @variable(model, p3 ∈ MOI.Parameter(100.0))
    @variable(model, x[i = 1:2], start = -i)
    @constraint(model, g_1, x[1]^2 <= p)
    @constraint(model, g_2, p * (x[1] + x[2])^2 <= p2)
    @objective(model, Min, (1 - x[1])^2 + p3 * (x[2] - x[1]^2)^2)
    return model, x, [g_1; g_2], [p; p2; p3]
end

function analytic_hessian(x, σ, μ, p)
    g_1_H = [2.0 0.0; 0.0 0.0]
    g_2_H = p[1] * [2.0 2.0; 2.0 2.0]
    f_H = zeros(2, 2)
    f_H[1, 1] = 2.0 + p[3] * 12.0 * x[1]^2 - p[3] * 4.0 * x[2]
    f_H[1, 2] = f_H[2, 1] = -p[3] * 4.0 * x[1]
    f_H[2, 2] = p[3] * 2.0
    return σ * f_H + μ' * [g_1_H, g_2_H]
end

function analytic_jacobian(x, p)
    g_1_J = [   
        2.0 * x[1], # ∂g_1/∂x_1
        0.0,       # ∂g_1/∂x_2
        -1.0,      # ∂g_1/∂p_1 
        0.0,      # ∂g_1/∂p_2
        0.0      # ∂g_1/∂p_3
    ]
    g_2_J = [
        p[1] * 2.0 * (x[1] + x[2]), # ∂g_2/∂x_1
        2.0 * (x[1] + x[2]),        # ∂g_2/∂x_2
        (x[1] + x[2])^2,            # ∂g_2/∂p_1
        -1.0,                        # ∂g_2/∂p_2
        0.0                         # ∂g_2/∂p_3
    ]
    return hcat(g_2_J, g_1_J)'[:,:]
end

function test_create_evaluator(model, x)
    @testset "Create NLP model" begin
        nlp, rows = create_nlp_model(model)
        @test nlp isa MOI.Nonlinear.Model
        @test rows isa Vector{ConstraintRef}
    end
    @testset "Create Evaluator" begin
        evaluator, rows = create_evaluator(model; x = x)
        @test evaluator isa MOI.Nonlinear.Evaluator
        @test rows isa Vector{ConstraintRef}
    end
end

function test_compute_optimal_hess_jacobian()
    @testset "Compute Optimal Hessian and Jacobian" begin
        # Model
        model, x, cons, params = create_nonlinear_jump_model()
        # Optimize
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Create evaluator
        test_create_evaluator(model, [x; params])
        evaluator, rows = create_evaluator(model; x = [x; params])
        # Compute Hessian and Jacobian
        num_var = length(x)
        full_hessian, full_jacobian = compute_optimal_hess_jac(evaluator, rows, [x; params])
        hessian = full_hessian[1:num_var, 1:num_var]
        # Check Hessian
        @test all(hessian .≈ analytic_hessian(value.(x), 1.0, dual.(cons), value.(params)))
        # TODO: Test hessial of parameters
        # Check Jacobian
        @test all(full_jacobian .≈ analytic_jacobian(value.(x), value.(params)))
    end
end

################################################
#=
# Test Derivatives and Sensitivity: QP problem 

From sIpopt paper: https://optimization-online.org/2011/04/3008/
=#

function create_nonlinear_jump_model_sipopt()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, p1 ∈ MOI.Parameter(4.5))
    @variable(model, p2 ∈ MOI.Parameter(1.0))
    @variable(model, x[i = 1:3] >= 0, start = -i)
    @constraint(model, g_1, 6 * x[1] + 3 * x[2] + 2 * x[3] - p1 == 0)
    @constraint(model, g_2, p2 * x[1] + x[2] - x[3] - 1 == 0)
    @objective(model, Min, x[1]^2 + x[2]^2 + x[3]^2)
    return model, x, [g_1; g_2], [p1; p2]
end

function test_compute_derivatives()
    @testset "Compute Derivatives No Inequalities" begin
        # Model
        model, primal_vars, cons, params = create_nonlinear_jump_model_sipopt()
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Analytical solutions case b
        pb = [4.5, 1.0]
        s_pb = [0.5, 0.5, 0.0]
        @assert all(isapprox(value.(primal_vars), s_pb; atol = 1e-6))
        # Analytical solutions case a
        pa = [5.0, 1.0]
        s_pa = [0.6327, 0.3878, 0.0204]
        set_parameter_value.(params, pa)
        optimize!(model)
        @assert is_solved_and_feasible(model)
        @assert all(isapprox(value.(primal_vars), s_pa; atol = 1e-4))
        # Compute derivatives without accounting for active set changes
        evaluator, rows = create_evaluator(model; x=[primal_vars; params])
        X, V_L, X_L, V_U, X_U, ineq_locations, has_up, has_low = compute_solution_and_bounds(primal_vars, rows)
        ∂s, K, N = compute_derivatives_no_relax(evaluator, rows, primal_vars, params, X, V_L, X_L, V_U, X_U, ineq_locations, has_up, has_low)
        # Check linear approx s_pb
        Δp = pb - pa
        s_pb_approx_violated = s_pa + ∂s[1:3, :] * Δp
        @test all(isapprox([0.5765; 0.3775; -0.0459], s_pb_approx_violated; atol = 1e-2))
        # Account for active set changes
        Δs = compute_derivatives(evaluator, rows, Δp; primal_vars, params)
        s_pb_approx = s_pa + Δs[1:3, :]
        @test all(isapprox(s_pb, s_pb_approx; atol = 1e-2))
    end
end

################################################
#=
# Test Sensitivity through finite differences
=#

function create_nonlinear_jump_model_1(p_val = [1.0; 2.0; 100])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x) 
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1]*sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective

    return model, [x; y], [con1; con2; con3], p
end

function eval_model_jump(model, primal_vars, cons, params, p_val)
    set_parameter_value.(params, p_val)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(primal_vars), dual.(cons), [dual.(LowerBoundRef(v)) for v in primal_vars if has_lower_bound(v)], [dual.(UpperBoundRef(v)) for v in primal_vars if has_upper_bound(v)]
end

function test_compute_derivatives_1()
    @testset "Compute Derivatives" begin
        # OPT Problem
        p_a = [1.0; 2.0; 100]
        model, primal_vars, cons, params = create_nonlinear_jump_model_1(p_a)
        # Debugging
        x_a, _λ_a, ν_La, ν_Ua = eval_model_jump(model, primal_vars, cons, params, p_a)
        ineq_locations = find_inequealities(cons)
        λ_a = deepcopy(_λ_a)
        λ_a[ineq_locations] = _λ_a[ineq_locations] .* -1
        s_a = [x_a; value.(get_slack_inequality.(cons[ineq_locations])); λ_a; ν_La; _λ_a[ineq_locations]; ν_Ua]
        # Compute derivatives
        p_b = [1.5; 2.00; 100.0]
        Δp = p_b - p_a
        (Δs, sp_approx), evaluator, cons = compute_derivatives(model, Δp; primal_vars, params)
        # Check solution
        x_b, _λ_b, ν_Lb, ν_Ub = eval_model_jump(model, primal_vars, cons, params, p_b)
        ineq_locations = find_inequealities(cons)
        λ_b = deepcopy(_λ_b)
        λ_b[ineq_locations] = _λ_b[ineq_locations] .* -1
        sp = [x_b; value.(get_slack_inequality.(cons[ineq_locations])); λ_b; ν_Lb; _λ_b[ineq_locations]; ν_Ub] 
        @test all(isapprox.(sp, sp_approx; atol = 1e-2))
        # Check derivatives using finite differences
        # ∂s_fd = FiniteDiff.finite_difference_jacobian((p) -> eval_model_jump(model, primal_vars, cons, params, p), p_a)
        # Δs_fd = ∂s_fd * Δp
    end
end