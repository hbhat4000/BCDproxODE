import autograd.numpy as np


def fitzhugh_nagumo_ode(t, x, params):
    '''
    This is the ODE function in eq.(17) of the paper.
    This function will be sent as a parameter to the scipy.integrate.ode().
    This function is called in the simulate.py file.

    Input:
        t: time: This should always be the first parameter.
            This is required by the scipy.integrate.ode().
        x: 2-dimensional state at time t.
        params: 3-dimensional parameters.

    Output:
        2-dimensional derivative dx/dt=[x0_dot, x1_dot].
    '''
    a, b, c = params[0], params[1], params[2]
    xdot = np.zeros_like(x)
    xdot[0] = c * (x[0] - ((x[0] ** 3) / 3) + x[1])
    xdot[1] = -(1 / c) * (x[0] - a + b * x[1])
    return xdot


def fitzhugh_nagumo_ode_vec(X, params):
    '''
    This is the ODE function in eq.(17) of the paper.
    The difference with fitzhugh__nagumo_ode: fitzhugh_nagumo_ode(t,x,params)
    takes a single state, but fitzhugh_nagumo_ode_vec(X,params)
    takes a set of states (in matrix X).
    For each row of the X, this function returns dx/dt.
    This function is used inside the autograd.
    We can achieve the output of this function by a loop over the
    fitzhugh_nagumo_ode(t, x, params), which is inefficient.

    Input:
        X: T*2 matrix of state.
            The i-th row shows the state at a specific time t_i.
        params: 3-dimensional parameters.

    Output:
        T*2 matrix of derivatives.
            The i-th row is the derivative of the i-th row of X.
    '''

    a, b, c = params[0], params[1], params[2]
    T = X.shape[0]
    t1 = (c * (X[0:T - 1, 0] - ((X[0:T - 1, 0] ** 3) / 3) + X[0:T - 1, 1]))
    t2 = (-(1 / c) * (X[0:T - 1, 0] - a + b * X[0:T - 1, 1]))
    terms = np.concatenate((t1.reshape(-1, 1), t2.reshape(-1, 1)), 1)

    return terms


