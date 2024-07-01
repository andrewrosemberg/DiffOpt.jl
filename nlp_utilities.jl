using JuMP.MOI
using SparseArrays

function create_nlp_model(model::JuMP.Model)
    rows = Any[]
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        if F <: VariableRef
            continue  # Skip variable bounds
        end
        for ci in all_constraints(model, F, S)
            push!(rows, ci)
            object = constraint_object(ci)
            MOI.Nonlinear.add_constraint(nlp, object.func, object.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    return nlp, rows
end

function fill_off_diagonal(H)
    ret = H + H'
    row_vals = SparseArrays.rowvals(ret)
    non_zeros = SparseArrays.nonzeros(ret)
    for col in 1:size(ret, 2)
        for i in SparseArrays.nzrange(ret, col)
            if col == row_vals[i]
                non_zeros[i] /= 2
            end
        end
    end
    return ret
end

function compute_optimal_hessian(evaluator, rows, x)
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), 1.0, dual.(rows))
    H = SparseArrays.sparse(I, J, V, length(x), length(x))
    return Matrix(fill_off_diagonal(H))
end

function compute_optimal_jacobian(evaluator, rows, x)
    jacobian_sparsity = MOI.jacobian_structure(evaluator)
    I = [i for (i, _) in jacobian_sparsity]
    J = [j for (_, j) in jacobian_sparsity]
    V = zeros(length(jacobian_sparsity))
    MOI.eval_constraint_jacobian(evaluator, V, value.(x))
    A = SparseArrays.sparse(I, J, V, length(rows), length(x))
    return Matrix(A)
end

function compute_optimal_hess_jac(evaluator, rows, x)
    hessian = compute_optimal_hessian(evaluator, rows, x)
    jacobian = compute_optimal_jacobian(evaluator, rows, x)
    
    return hessian, jacobian
end

all_primal_vars(model::Model) = filter(x -> !is_parameter(x), all_variables(model))
all_params(model::Model) = filter(x -> is_parameter(x), all_variables(model))

function create_evaluator(model::Model; x=all_variables(model))
    nlp, rows = create_nlp_model(model)
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
    MOI.initialize(evaluator, [:Hess, :Jac])
    return evaluator, rows
end

function compute_derivatives(evaluator, cons; primal_vars=all_primal_vars(model), params=all_params(model)
)
    # Setting
    num_vars = length(primal_vars)
    num_parms = length(params)
    num_cons = length(cons)
    all_vars = [primal_vars; params]

    # Primal solution
    X = diagm(value.(primal_vars))

    # Dual of the bounds
    bound_duals = zeros(length(primal_vars))
    for i in 1:length(primal_vars)
        if has_lower_bound(primal_vars[i])
            bound_duals[i] = dual.(LowerBoundRef(primal_vars[i]))
        end
        if has_upper_bound(primal_vars[i])
            bound_duals[i] -= dual.(UpperBoundRef(primal_vars[i]))
        end
    end
    V = diagm(bound_duals)

    # Function Derivatives
    hessian, jacobian = compute_optimal_hess_jac(evaluator, cons, all_vars)

    # Hessian of the lagrangian wrt the primal variables
    W = hessian[1:num_vars, 1:num_vars]
    # Jacobian of the constraints wrt the primal variables
    A = jacobian[:, 1:num_vars]
    # Partial second derivative of the lagrangian wrt primal solution and parameters
    ∇ₓₚL = hessian[num_vars+1:end, 1:num_vars]
    # Partial derivative of the equality constraintswith wrt parameters
    ∇ₚC = jacobian[:, num_vars+1:end]

    # M matrix
    M = zeros(num_vars * 2 + num_cons, num_vars * 2 + num_cons)

    # M = [
    #     [W A' -I];
    #     [A 0 0];
    #     [V 0 X]
    # ]

    M[1:num_vars, 1:num_vars] = W
    M[1:num_vars, num_vars+1:num_vars+num_cons] = A'
    M[num_vars+1:num_vars+num_cons, 1:num_vars] = A
    M[1:num_vars, num_vars+num_cons+1:end] = -I(num_vars)
    M[num_vars+num_cons+1:end, 1:num_vars] = V
    M[num_vars+num_cons+1:end, num_vars+num_cons+1:end] = X

    # N matrix
    N = [∇ₓₚL ; ∇ₚC; zeros(num_vars, num_parms)]

    # sesitivity of the solution (primal-dual_constraints-dual_bounds) with respect to the parameters
    return pinv(M) * N
end

function compute_derivatives(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))
    evaluator, rows = create_evaluator(model)
    return compute_derivatives(evaluator, rows; primal_vars=primal_vars, params=params), evaluator, rows
end
