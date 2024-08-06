using JuMP
using Ipopt
using Test
using FiniteDiff

include("nlp_utilities.jl")

################################################
#=
# Test JuMP Hessian and Jacobian

From JuMP Tutorial for Querying Hessians:
https://github.com/jump-dev/JuMP.jl/blob/301d46e81cb66c74c6e22cd89fb89ced740f157b/docs/src/tutorials/nonlinear/querying_hessians.jl#L67-L72
=#
################################################

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
        @test all(hessian .≈ analytic_hessian(value.(x), -1.0, dual.(cons), value.(params)))
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
################################################

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
        s_pb = [0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        @assert all(isapprox.([value.(primal_vars); dual.(cons); dual.(LowerBoundRef.(primal_vars))], s_pb; atol = 1e-6))
        # Analytical solutions case a
        pa = [5.0, 1.0]
        s_pa = [0.6327, 0.3878, 0.0204, 0.1633, 0.2857, 0, 0, 0]
        set_parameter_value.(params, pa)
        optimize!(model)
        @assert is_solved_and_feasible(model)
        @assert all(isapprox.([value.(primal_vars); dual.(cons); dual.(LowerBoundRef.(primal_vars))], s_pa; atol = 1e-4))
        # Compute derivatives without accounting for active set changes
        evaluator, rows = create_evaluator(model; x=[primal_vars; params])
        X, V_L, X_L, V_U, X_U, leq_locations, geq_locations, ineq_locations, has_up, has_low = compute_solution_and_bounds(primal_vars, rows)
        ∂s, K, N = compute_derivatives_no_relax(evaluator, rows, primal_vars, params, X, V_L, X_L, V_U, X_U, leq_locations, geq_locations, ineq_locations, has_up, has_low)
        # Check linear approx s_pb
        Δp = pb - pa
        s_pb_approx_violated = s_pa + ∂s * Δp
        @test all(isapprox.([0.5765; 0.3775; -0.0459; 0.1327; 0.3571; 0.0; 0.0; 0.0], s_pb_approx_violated; atol = 1e-4))
        # Account for active set changes
        Δs, s_pb_approx = compute_sensitivity(evaluator, rows, Δp; primal_vars, params)
        # s_pb_approx = s_pa .+ Δs
        @test all(isapprox.([0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], s_pb_approx; atol = 1e-6))
    end
end

################################################
#=
# Test Sensitivity through analytical
=#
################################################

function create_jump_model_1(p_val = [1.5])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x) 

    # Constraints
    @constraint(model, con1, x >= p)
    @constraint(model, con2, x >= 2)
    @objective(model, Min, x^2)

    return model, [x], [con1; con2], [p]
end

function create_jump_model_2(p_val = [1.5])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x >= 2.0) 

    # Constraints
    @constraint(model, con1, x >= p)
    @objective(model, Min, x^2)

    return model, [x], [con1], [p]
end

function create_jump_model_3(p_val = [-1.5])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x) 

    # Constraints
    @constraint(model, con1, x <= p)
    @constraint(model, con2, x <= -2)
    @objective(model, Min, -x)

    return model, [x], [con1; con2], [p]
end


DICT_PROBLEMS_Analytical = Dict(
    "geq no impact" => (p_a=[1.5], Δp=[0.2], Δs_a=[0.0; -0.2; 0.0; 0.0; 0.0; 0.0; 0.0], model_generator=create_jump_model_1),
    "geq active constraint change" => (p_a=[1.9], Δp=[0.2], Δs_a=[0.1; -0.1; 0.1], model_generator=create_jump_model_1),
    "geq impact" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.2; 0.0; 0.2; 0.4; 0.0; 0.4; 0.0], model_generator=create_jump_model_1),
    "geq active bound change" => (p_a=[2.1], Δp=[-0.2], Δs_a=[-0.1; 0.1], model_generator=create_jump_model_2),
    "geq bound impact" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.2; 0.0; 0.4; 0.0; 0.4], model_generator=create_jump_model_2),
    "leq no impact" => (p_a=[-1.5], Δp=[-0.2], Δs_a=[0.0; 0.2; 0.0; 0.0; 0.0; 0.0; 0.0], model_generator=create_jump_model_3),
    "leq active constraint change" => (p_a=[-1.9], Δp=[-0.2], Δs_a=[-0.1; 0.1; -0.1], model_generator=create_jump_model_3),
)

function test_compute_derivatives_Analytical()
    @testset "Compute Derivatives Analytical: $problem_name" for (problem_name, (p_a, Δp, Δs_a, model_generator)) in DICT_PROBLEMS_Analytical
        # OPT Problem
        model, primal_vars, cons, params = model_generator()
        eval_model_jump(model, primal_vars, cons, params, p_a)
        # Compute derivatives
        (Δs, sp_approx), evaluator, cons = compute_sensitivity(model, Δp; primal_vars, params)
        # Check sensitivities
        @test all(isapprox.(Δs[1:length(Δs_a)], Δs_a; atol = 1e-6))
    end
end

################################################
#=
# Test Sensitivity through finite differences
=#
################################################

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

function create_nonlinear_jump_model_2(p_val = [3.0; 2.0; 10])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x <= 10) 
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1]*sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_3(p_val = [3.0; 2.0; 10])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x <= 10) 
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1]*sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_4(p_val = [1.0; 2.0; 100])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x) 
    @variable(model, y)

    # Constraints
    @constraint(model, con0, x == p[1] - 0.5)
    @constraint(model, con1, y >= p[1]*sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective

    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_5(p_val = [1.0; 2.0; 100])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x) 
    @variable(model, y)

    # Constraints
    fix(x, 0.5)
    con0 = JuMP.FixRef(x)
    @constraint(model, con1, y >= p[1]*sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective

    return model, [x; y], [con0; con1; con2; con3], p
end

function create_nonlinear_jump_model_6(p_val = [100.0; 200.0])
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Parameters
    @variable(model, p[i=1:2] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x[i=1:2])
    @variable(model, z) # >= 2.0)
    @variable(model, w) # <= 3.0)
    # @variable(model, f[1:2])

    # Constraints
    @constraint(model, con1, x[2] - 0.0001 * x[1]^2 - 0.2 * z^2 - 0.3 * w^2 >= p[1] + 1)
    @constraint(model, con2,  x[1] + 0.001 * x[2]^2 + 0.5 * w^2 + 0.4 * z^2 <= 10 * p[1] + 2)
    @constraint(model, con3, z^2 + w^2 == 13)
    @objective(model, Min, x[2] - x[1] + z - w)

    return model, [x; z; w], [con2; con3], p
end

function eval_model_jump(model, primal_vars, cons, params, p_val)
    set_parameter_value.(params, p_val)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(primal_vars), dual.(cons), [dual.(LowerBoundRef(v)) for v in primal_vars if has_lower_bound(v)], [dual.(UpperBoundRef(v)) for v in primal_vars if has_upper_bound(v)]
end

function stack_solution(cons, leq_locations, geq_locations, x, _λ, ν_L, ν_U)
    ineq_locations = vcat(geq_locations, leq_locations)
    return Float64[x; value.(get_slack_inequality.(cons[ineq_locations])); _λ; ν_L; _λ[geq_locations]; ν_U; _λ[leq_locations]]
end

function print_wrong_sensitive(Δs, Δs_fd, primal_vars, cons, leq_locations, geq_locations)
    ineq_locations = vcat(geq_locations, leq_locations)
    println("Some sensitivities are not correct: \n")
    # primal vars
    num_primal_vars = length(primal_vars)
    for (i, v) in enumerate(primal_vars)
        if !isapprox(Δs[i], Δs_fd[i]; atol = 1e-6)
            println("Primal var: ", v, " | Δs: ", Δs[i], " | Δs_fd: ", Δs_fd[i])
        end
    end
    # slack vars
    num_slack_vars = length(ineq_locations)
    num_w = num_slack_vars + num_primal_vars
    for (i, c) in enumerate(cons[ineq_locations])
        if !isapprox(Δs[i + num_primal_vars], Δs_fd[i + num_primal_vars] ; atol = 1e-6)
            println("Slack var: ", c, " | Δs: ", Δs[i + num_primal_vars], " | Δs_fd: ", Δs_fd[i + num_primal_vars])
        end
    end
    # dual vars
    num_cons = length(cons)
    for (i, c) in enumerate(cons)
        if !isapprox(Δs[i + num_w], Δs_fd[i + num_w] ; atol = 1e-6)
            println("Dual var: ", c, " | Δs: ", Δs[i + num_w], " | Δs_fd: ", Δs_fd[i + num_w])
        end
    end
    # dual lower bound primal vars
    var_lower = [v for v in primal_vars if has_lower_bound(v)]
    num_lower_bounds = length(var_lower)
    for (i, v) in enumerate(var_lower)
        if !isapprox(Δs[i + num_w + num_cons], Δs_fd[i + num_w + num_cons] ; atol = 1e-6)
            lower_bound_ref = LowerBoundRef(v)
            println("lower bound dual: ", lower_bound_ref, " | Δs: ", Δs[i + num_w + num_cons], " | Δs_fd: ", Δs_fd[i + num_w + num_cons])
        end
    end
    # dual lower bound slack vars
    for (i, c) in enumerate(cons[geq_locations])
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds], Δs_fd[i + num_w + num_cons + num_lower_bounds] ; atol = 1e-6)
            println("lower bound slack dual: ", c, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds])
        end
    end
    for (i, c) in enumerate(cons[leq_locations])
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds + length(geq_locations)], Δs_fd[i + num_w + num_cons + num_lower_bounds + length(geq_locations)] ; atol = 1e-6)
            println("upper bound slack dual: ", c, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds + length(geq_locations)], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds + length(geq_locations)])
        end
    end
    # dual upper bound primal vars
    var_upper = [v for v in primal_vars if has_upper_bound(v)]
    for (i, v) in enumerate(var_upper)
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds + num_slack_vars], Δs_fd[i + num_w + num_cons + num_lower_bounds + num_slack_vars] ; atol = 1e-6)
            upper_bound_ref = UpperBoundRef(v)
            println("upper bound dual: ", upper_bound_ref, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds + num_slack_vars], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds + num_slack_vars])
        end
    end
end

DICT_PROBLEMS = Dict(
    "QP_JuMP" => (p_a=[1.0; 2.0; 100.0], Δp=[-0.5; 0.5; 0.1], model_generator=create_nonlinear_jump_model),
    "QP_sIpopt" => (p_a=[4.5; 1.0], Δp=[0.001; 0.0], model_generator=create_nonlinear_jump_model_sipopt),
    "QP_sIpopt" => (p_a=[5.0; 1.0], Δp=[-0.5; 0.0], model_generator=create_nonlinear_jump_model_sipopt),
    "NLP_1" => (p_a=[3.0; 2.0; 200], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_2" => (p_a=[3.0; 2.0; 200], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_3" => (p_a=[3.0; 2.0; 200], Δp=[0.0; 0.0; 0.001], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_4" => (p_a=[3.0; 2.0; 200], Δp=[0.1; 0.5; 0.5], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_4" => (p_a=[3.0; 2.0; 200], Δp=[0.5; -0.5; 0.1], model_generator=create_nonlinear_jump_model_1),
    "NLP_2" => (p_a=[3.0; 2.0; 10], Δp=[0.01; 0.0; 0.0], model_generator=create_nonlinear_jump_model_2),
    "NLP_2_2" => (p_a=[3.0; 2.0; 10], Δp=[-0.1; 0.0; 0.0], model_generator=create_nonlinear_jump_model_2),
    "NLP_3" => (p_a=[3.0; 2.0; 10], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_2" => (p_a=[3.0; 2.0; 10], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_3" => (p_a=[3.0; 2.0; 10], Δp=[0.0; 0.0; 0.001], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_4" => (p_a=[3.0; 2.0; 10], Δp=[0.5; 0.001; 0.5], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_5" => (p_a=[3.0; 2.0; 10], Δp=[0.1; 0.3; 0.1], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_6" => (p_a=[3.0; 2.0; 10], Δp=[0.1; 0.2; -0.5], model_generator=create_nonlinear_jump_model_3),
    "NLP_4" => (p_a=[1.0; 2.0; 100], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_4),
    "NLP_5" => (p_a=[1.0; 2.0; 100], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_5),
    "NLP_6" => (p_a=[100.0; 200.0], Δp=[0.2; 0.5], model_generator=create_nonlinear_jump_model_6),
)

function test_compute_derivatives_Finite_Diff()
    # @testset "Compute Derivatives: $problem_name" 
    for (problem_name, (p_a, Δp, model_generator)) in DICT_PROBLEMS
        # OPT Problem
        model, primal_vars, cons, params = model_generator()
        eval_model_jump(model, primal_vars, cons, params, p_a)

        println("$problem_name: ", model)
        # Compute derivatives
        # Δp = [0.001; 0.0; 0.0]
        p_b = p_a .+ Δp
        (Δs, sp_approx), evaluator, cons = compute_sensitivity(model, Δp; primal_vars, params)
        leq_locations, geq_locations = find_inequealities(cons)
        # Check derivatives using finite differences
        ∂s_fd = FiniteDiff.finite_difference_jacobian((p) -> stack_solution(cons, leq_locations, geq_locations, eval_model_jump(model, primal_vars, cons, params, p)...), p_a)
        Δs_fd = ∂s_fd * Δp
        # actual solution
        sp = stack_solution(cons, leq_locations, geq_locations, eval_model_jump(model, primal_vars, cons, params, p_b)...)
        num_important = length(primal_vars)
        # Check sensitivities
        if all(isapprox.(Δs, Δs_fd; rtol = 1e-3, atol=14-6)) || all(isapprox.(sp[1:num_important], sp_approx[1:num_important]; rtol = 1e-3, atol=14-6))
            println("All sensitivities are correct")
        else
            @show Δp 
            print_wrong_sensitive(Δs, Δs_fd, primal_vars, cons, leq_locations, geq_locations)
            # Check solution
            println(all(isapprox.(sp[1:num_important], sp_approx[1:num_important]; atol = 1e-6)))
        end
        println("--------------------")
    end
end