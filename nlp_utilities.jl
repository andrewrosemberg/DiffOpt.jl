using JuMP.MOI
using SparseArrays

"""
    create_nlp_model(model::JuMP.Model)

Create a Nonlinear Programming (NLP) model from a JuMP model.
"""
function create_nlp_model(model::JuMP.Model)
    rows = Vector{ConstraintRef}(undef, 0)
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

"""
    fill_off_diagonal(H)

Filling the off-diagonal elements of a sparse matrix to make it symmetric.
"""
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

"""
    compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Hessian of the Lagrangian.
"""
function compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    MOI.eval_hessian_lagrangian(evaluator, V, value.(x), 1.0, dual.(rows))
    H = SparseArrays.sparse(I, J, V, length(x), length(x))
    return Matrix(fill_off_diagonal(H))
end

"""
    compute_optimal_jacobian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Jacobian of the constraints.
"""
function compute_optimal_jacobian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    jacobian_sparsity = MOI.jacobian_structure(evaluator)
    I = [i for (i, _) in jacobian_sparsity]
    J = [j for (_, j) in jacobian_sparsity]
    V = zeros(length(jacobian_sparsity))
    MOI.eval_constraint_jacobian(evaluator, V, value.(x))
    A = SparseArrays.sparse(I, J, V, length(rows), length(x))
    return Matrix(A)
end

"""
    compute_optimal_hess_jac(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})

Compute the optimal Hessian of the Lagrangian and Jacobian of the constraints.
"""
function compute_optimal_hess_jac(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{ConstraintRef}, x::Vector{VariableRef})
    hessian = compute_optimal_hessian(evaluator, rows, x)
    jacobian = compute_optimal_jacobian(evaluator, rows, x)
    
    return hessian, jacobian
end

"""
    all_primal_vars(model::Model)

Get all the primal variables in the model.
"""
all_primal_vars(model::Model) = filter(x -> !is_parameter(x), all_variables(model))

"""
    all_params(model::Model)

Get all the parameters in the model.
"""
all_params(model::Model) = filter(x -> is_parameter(x), all_variables(model))

"""
    create_evaluator(model::Model; x=all_variables(model))

Create an evaluator for the model.
"""
function create_evaluator(model::Model; x=all_variables(model))
    nlp, rows = create_nlp_model(model)
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(nlp, backend, index.(x))
    MOI.initialize(evaluator, [:Hess, :Jac])
    return evaluator, rows
end

"""
    is_inequality(con::ConstraintRef)

Check if the constraint is an inequality.
"""
function is_inequality(con::ConstraintRef)
    set_type = typeof(MOI.get(owner_model(con), MOI.ConstraintSet(), con))
    return set_type <: MOI.LessThan || set_type <: MOI.GreaterThan
end

"""
    find_inequealities(cons::Vector{ConstraintRef})

Find the indices of the inequality constraints.
"""
function find_inequealities(cons::Vector{ConstraintRef})
    ineq_locations = zeros(length(cons))
    for i in 1:length(cons)
        if is_inequality(cons[i])
            ineq_locations[i] = true
        end
    end
    return findall(x -> x ==1, ineq_locations)
end

"""
    get_slack_inequality(con::ConstraintRef)

Get the reference to the canonical function that is equivalent to the slack variable of the inequality constraint.
"""
function get_slack_inequality(con::ConstraintRef)
    slack = MOI.get(owner_model(con), CanonicalConstraintFunction(), con)
    return slack
end

"""
    compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}; primal_vars::Vector{VariableRef}=all_primal_vars(model), params::Vector{VariableRef}=all_params(model))

Compute the derivatives of the solution with respect to the parameters.
"""
function compute_derivatives(evaluator::MOI.Nonlinear.Evaluator, cons::Vector{ConstraintRef}; 
    primal_vars::Vector{VariableRef}, params::Vector{VariableRef}
)
    @assert all(x -> is_parameter(x), params) "All parameters must be parameters"

    # Setting
    num_vars = length(primal_vars)
    num_parms = length(params)
    num_cons = length(cons)
    ineq_locations = find_inequealities(cons)
    num_ineq = length(ineq_locations)
    slack_vars = [get_slack_inequality(cons[i]) for i in ineq_locations]
    all_vars = [primal_vars; params]

    # Primal solution
    X = diagm(value.([primal_vars; slack_vars]))

    # Dual of the bounds
    bound_duals = zeros(num_vars+num_ineq)
    for i in 1:num_vars
        if has_lower_bound(primal_vars[i])
            bound_duals[i] = dual.(LowerBoundRef(primal_vars[i]))
        end
        if has_upper_bound(primal_vars[i])
            bound_duals[i] -= dual.(UpperBoundRef(primal_vars[i]))
        end
    end
    for (i, con) in enumerate(cons[ineq_locations])
        bound_duals[num_vars+i] = dual.(con)
    end
    V = diagm(bound_duals)

    # Function Derivatives
    hessian, jacobian = compute_optimal_hess_jac(evaluator, cons, all_vars)

    # Hessian of the lagrangian wrt the primal variables
    W = zeros(num_vars + num_ineq, num_vars + num_ineq)
    W[1:num_vars, 1:num_vars] = hessian[1:num_vars, 1:num_vars]
    # Jacobian of the constraints wrt the primal variables
    A = zeros(num_cons, num_vars + num_ineq)
    A[:, 1:num_vars] = jacobian[:, 1:num_vars]
    for (i,j) in enumerate(ineq_locations)
        A[j, num_vars+i] = 1
    end
    # Partial second derivative of the lagrangian wrt primal solution and parameters
    # TODO Fix dimensions
    ∇ₓₚL = zeros(num_parms, num_vars + num_ineq)
    ∇ₓₚL[:, 1:num_vars] = hessian[num_vars+1:end, 1:num_vars]
    # Partial derivative of the equality constraintswith wrt parameters
    ∇ₚC = jacobian[:, num_vars+1:end]

    # M matrix
    M = zeros(2 * (num_vars + num_ineq) + num_cons, 2 * (num_vars + num_ineq) + num_cons)

    # M = [
    #     [W A' -I];
    #     [A 0 0];
    #     [V 0 X]
    # ]

    M[1:num_vars + num_ineq, 1:num_vars + num_ineq] = W
    M[1:num_vars + num_ineq, num_vars + num_ineq + 1 : num_vars + num_ineq + num_cons] = A'
    M[num_vars + num_ineq+1:num_vars + num_ineq+num_cons, 1:num_vars + num_ineq] = A
    M[1:num_vars + num_ineq, num_vars + num_ineq+num_cons+1:end] = -I(num_vars + num_ineq)
    M[num_vars + num_ineq+num_cons+1:end, 1:num_vars + num_ineq] = V
    M[num_vars + num_ineq+num_cons+1:end, num_vars + num_ineq+num_cons+1:end] = X

    # N matrix
    N = [∇ₓₚL ; ∇ₚC; zeros(num_vars + num_ineq, num_parms)]

    # Sesitivity of the solution (primal-dual_constraints-dual_bounds) with respect to the parameters
    return pinv(M) * N
end

"""
    compute_derivatives(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))

Compute the derivatives of the solution with respect to the parameters.
"""
function compute_derivatives(model::Model; primal_vars=all_primal_vars(model), params=all_params(model))
    evaluator, rows = create_evaluator(model; x=[primal_vars; params])
    return compute_derivatives(evaluator, rows; primal_vars=primal_vars, params=params), evaluator, rows
end
