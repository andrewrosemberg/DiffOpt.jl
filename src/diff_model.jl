"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note 
Currently supports differentiating linear and quadratic programs only.
"""
function diff_model(_model::MOI.AbstractOptimizer)
    
    model = deepcopy(_model)

    Q, q, G, h, A, b, nz, var_idx, nineq, ineq_con_idx, neq, eq_con_idx = get_problem_data(model)
    
    z = zeros(0) # solution
    λ = zeros(0) # lagrangian variables
    ν = zeros(0)

    """
        Solving the convex optimization problem in forward pass
    """
    function forward()
        # solve the model
        MOI.optimize!(model)
        
        # check status
        @assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        
        # get and save the solution
        z = MOI.get(model, MOI.VariablePrimal(), var_idx)
        
        # get and save dual variables
        λ = MOI.get(model, MOI.ConstraintDual(), ineq_con_idx)
    
        if neq > 0
            ν = MOI.get(model, MOI.ConstraintDual(), eq_con_idx)
        end
    
        return z
    end
    
    """
        Method to differentiate and obtain gradients/jacobians
        of z, λ, ν  with respect to the parameters specified in
        in argument
    """
    function backward(params)
        grads = []
        LHS = create_LHS_matrix(z, λ, Q, G, h, A)
        for param in params
            if param == "h"
                RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
                                        λ, zeros(nineq, nz), ones(nineq,1),
                                        ν, zeros(neq, nz), zeros(neq, 1))
                push!(grads, LHS \ RHS)
            elseif param == "Q"
                RHS = create_RHS_matrix(z, ones(nz, nz), zeros(nz, 1),
                                        λ, zeros(nineq, nz), zeros(nineq,1),
                                        ν, zeros(neq, nz), zeros(neq, 1))
                push!(grads, LHS \ RHS)
            else
                push!(grads, [])
            end
        end
        return grads
    end
    
    () -> (forward, backward)
end