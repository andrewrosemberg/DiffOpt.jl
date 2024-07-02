using JuMP
using Ipopt
using Test

function create_nonlinear_jump_model()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    # Parameters
    @variable(model, p ∈ MOI.Parameter(1.0))
    @variable(model, p2 ∈ MOI.Parameter(2.0))
    @variable(model, p3 ∈ MOI.Parameter(100.0))
    @variable(model, x[i = 1:2], start = -i)
    @constraint(model, g_1, x[1]^2 <= p)
    @constraint(model, g_2, p * (x[1] + x[2])^2 <= p2)
    @objective(model, Min, (1 - x[1])^2 + p3 * (x[2] - x[1]^2)^2)
    return model, x, [g_1; g_2], [p; p2; p3]
end

function analytic_hessian(x, σ, μ)
    g_1_H = [2.0 0.0; 0.0 0.0]
    g_2_H = [2.0 2.0; 2.0 2.0]
    f_H = zeros(2, 2)
    f_H[1, 1] = 2.0 + 1200.0 * x[1]^2 - 400.0 * x[2]
    f_H[1, 2] = f_H[2, 1] = -400.0 * x[1]
    f_H[2, 2] = 200.0
    return σ * f_H + μ' * [g_1_H, g_2_H]
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
        num_var = length(x)
        test_create_evaluator(model, x)
        evaluator, rows = create_evaluator(model; x = [x; params])
        # Optimize
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Compute Hessian and Jacobian
        full_hessian, full_jacobian = compute_optimal_hess_jac(evaluator, rows, [x; params])
        hessian = full_hessian[1:num_var, 1:num_var]
        # Check Hessian
        @test hessian .≈ analytic_hessian(value.(x), 1.0, dual.(cons))
        # TODO: Check Jacobian
    end
end

# TODO: Test derivatives

function test_compute_derivatives()
    model, x = create_nonlinear_jump_model()
    optimize!(model)
    @assert is_solved_and_feasible(model)

end