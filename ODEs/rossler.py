import autograd.numpy as np

def rossler_ode(t, x, params):
    '''
    This is the ODE function in eq.(18).
    This function will be sent as a parameter to the scipy.integrate.ode().
    This function is called in the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
            This is required by the scipy.integrate.ode().
        x: 3-dimensional state at time t.
        params: 3-dimensional parameters.

    Output:
        3-dimensional derivative dx/dt=[x0_dot, x1_dot, x2_dot].
    '''

    a, b, c = params[0], params[1], params[2]
    xdot = np.zeros_like(x)
    xdot[0] = -x[1] - x[2]
    xdot[1] = x[0] + a*x[1]
    xdot[2] = b + (x[2] * (x[0] - c) )
    return xdot

def rossler_ode_vec(X, params):
    '''
    This is the ODE function in eq.(18) of the paper.
    The difference with rossler_ode: rossler_ode(t,x,params) takes a single
    state, but rossler_ode_vec(X,params)takes a set of states (in matrix X).
    For each row of the X, this function returns dx/dt.
    This function is used inside the autograd.
    We can achieve the output of this function by a loop over the
    fitzhugh_ode(t, x, params), which is inefficient.

    Input:
        X: T*3 matrix of state.
            The i-th row shows the state at a specific time t_i.
        params: 3-dimensional parameters.

    Output:
        T*3 matrix of derivatives.
            The i-th row is the derivative of the i-th row of X.
    '''
    a, b, c = params[0], params[1], params[2]
    T = X.shape[0]
    t1 = -X[0:T-1,1] - X[0:T-1,2]
    t2 = X[0:T-1,0] + a*X[0:T-1,1]
    t3 = b + (X[0:T-1,2] * (X[0:T-1,0] - c) )
    terms = np.concatenate((t1.reshape(-1, 1), t2.reshape(-1, 1),
                            t3.reshape(-1, 1)), 1)
    return terms