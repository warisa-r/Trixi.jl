abstract type AbstractController end

# This comes later after we test out other controllers and found out that a controller is superior to others.

@inline function stepsize_controller!(integrator, alg)
    stepsize_controller!(integrator, integrator.opts.controller, alg)
end

# checks whether the controller should accept a step based on the error estimate
@inline function accept_step_controller(integrator, controller::AbstractController)
    return integrator.EEst <= 1
end

@inline function step_accept_controller!(integrator, alg, q)
    step_accept_controller!(integrator, integrator.opts.controller, alg, q)
end

@inline function step_reject_controller!(integrator, alg)
    step_reject_controller!(integrator, integrator.opts.controller, alg)
end

# Implement the PID controller in such way that it can be used with the PERK schemes
struct PIDController <: AbstractController
    beta::Vector{Float64}      # beta parameter of the PID controller
    err::Vector{Float64} # history of the error estimates
    accept_safety::Float64   # accept a step if the predicted change of the step size
    # is bigger than this parameter
    limiter::Function    # limiter of the dt factor (before clipping)
end

function PIDController(beta1, beta2, beta3;
                       limiter = default_dt_factor_limiter,
                       accept_safety = 0.81) # 0.81 - 0.9^2
    beta = [float(beta1), float(beta2), float(beta3)]  # Create a Vector{Float64} of size 3
    err = ones(Float64, 3)
    return PIDController(beta, err, accept_safety, limiter)
end

default_dt_factor_limiter(x) = one(x) + atan(x - one(x))

# Taken from OrdinaryDiffEq.jl
# This funtion is to calculate the factor to multiply the current time step by
function stepsize_controller!(integrator, controller::PIDController, alg)
    beta1, beta2, beta3 = controller.beta

    EEst = integrator.EEst

    # If the error estimate is zero, we can increase the step size as much as
    # desired. This additional check fixes problems of the code below when the
    # error estimates become zero
    # -> err1, err2, err3 become Inf
    # -> err1^positive_number * err2^negative_number becomes NaN
    # -> dt becomes NaN
    #
    # `EEst_min` is smaller than PETSC_SMALL used in the equivalent logic in PETSc.
    # For example, `eps(Float64) ≈ 2.2e-16` but `PETSC_SMALL ≈ 1.0e-10` for `double`.
    EEst_min = eps(Float64)
    # The code below is a bit more robust than
    # ```
    # if iszero(EEst)
    #   EEst = eps(typeof(EEst))
    # end
    # ```
    EEst = ifelse(EEst > EEst_min, EEst, EEst_min)

    controller.err[1] = inv(EEst)
    err1, err2, err3 = controller.err

    k = 3 # the order of the main method of EmbeddedPairedExplicitRK3
    dt_factor = err1^(beta1 / k) * err2^(beta2 / k) * err3^(beta3 / k)
    if isnan(dt_factor)
        @warn "unlimited dt_factor" dt_factor err1 err2 err3 beta1 beta2 beta3 k
    end
    dt_factor = controller.limiter(dt_factor)

    # Note: No additional limiting of the form
    #   dt_factor = max(qmin, min(qmax, dt_factor))
    # is necessary since the `limiter` should take care of that. The default limiter
    # ensures
    #   0.21 ≈ limiter(0) <= dt_factor <= limiter(Inf) ≈ 2.57
    # See Söderlind, Wang (2006), Section 6.
    integrator.qold = dt_factor
    integrator.dtpropose = integrator.dt * dt_factor
    return dt_factor
end

# Instead of the criteria in the paper of embedded RK21, we use this criteria instead.
function accept_step_controller(integrator, controller::PIDController)
    return integrator.qold >= controller.accept_safety
end

# This function is to actually handles the controller and reset the next default time step to dt_opt
function step_accept_controller!(integrator, controller::PIDController, alg, dt_factor)
    @unpack qsteady_min, qsteady_max = integrator.opts

    # We will deal with this later

    if qsteady_min <= inv(dt_factor) <= qsteady_max
        dt_factor = one(dt_factor)
    end

    @inbounds begin
        controller.err[3] = controller.err[2]
        controller.err[2] = controller.err[1]
    end

    # We don't have to update dt or t or u anymore since step! already did the job

    # Set dt to the optimum value from the stability polynomial optimization
    return integrator.dt * dt_factor # new dt
end

function step_reject_controller!(integrator, controller::PIDController, alg, u_old)
    integrator.nreject += 1 # Increment nreject
    integrator.dt *= integrator.qold # Set the proposed time step with regard to the time controller
end
