import autograd.numpy as np


def lorenz96_ode(t, x, params):
    '''
    This is the ODE function in eq.(19) of the paper.
    This function will be sent as a parameter to the scipy.integrate.ode().
    This function is called in the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
            This is required by the scipy.integrate.ode().
        x: d-dimensional state at time t.
        params: 1-dimensional parameter.

    Output:
        d-dimensional derivative dx/dt=[x0_dot, x1_dot,...].
    '''

    F = params[0]
    T = x.shape[0]
    xdot = np.zeros_like(x)
    xdot[0] = ((x[1] - x[T - 2]) * x[T - 1]) - x[0]
    xdot[1] = ((x[2] - x[T - 1]) * x[0]) - x[1]
    xdot[T - 1] = ((x[0] - x[T - 3]) * x[T - 2]) - x[T - 1]
    for i in range(2, T - 1):
        xdot[i] = ((x[i + 1] - x[i - 2]) * x[i - 1]) - x[i]
    xdot = xdot + F
    return xdot

def lorenz96_ode_vec(X, params):
    '''
    This is the ODE function in eq.(19) of the paper.
    The difference with lorenz96_ode: lorenz96_ode(t,x,params) takes a single
    state, but lorenz96_ode_vec(X,params) takes a set of states (in matrix X).
    For each row of the X, this function returns dx/dt.
    This function is used inside the autograd.
    We can achieve the output of this function by a loop over the
    lorenz96_ode(t, x, params), which is inefficient.

    Input:
        X: T*d matrix of state.
            The i-th row shows the state at a specific time t_i.
        params: 1-dimensional parameters.

    Output:
        T*d matrix of derivatives.
            The i-th row is the derivative of the i-th row of X.
    '''

    F = params[0]
    T,d=X.shape
    temp1 = ((X[0:T - 1, 3:d] - X[0:T - 1, 0:d - 3]) * X[0:T - 1, 1:d - 2] -
             X[0:T - 1, 2:d - 1] + F)
    temp2 = ((X[0:T - 1, 1] - X[0:T - 1, d - 2]) * X[0:T - 1, d - 1] -
             X[0:T - 1, 0] + F)
    temp3 = ((X[0:T - 1, 2] - X[0:T - 1, d - 1]) * X[0:T - 1, 0] -
             X[0:T - 1, 1] + F)
    temp4 = ((X[0:T - 1, 0] - X[0:T - 1, d - 3]) * X[0:T - 1, d - 2] -
             X[0:T - 1, d - 1] + F)
    terms = np.concatenate((temp2.reshape(-1, 1), temp3.reshape(-1, 1),
                            temp1, temp4.reshape(-1, 1)), 1)

    return terms