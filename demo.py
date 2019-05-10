import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('./AUX/')
from fit_direct import fit_direct
from simulate import *

# Set ODE_str to 'fitzhugh_nagumo' or 'lotka_volterra' or 'rossler' or 'lorenz96'
ODE_str = 'fitzhugh_nagumo' # it takes 30-40 seconds
#ODE_str = 'rossler' # it takes 30-40 seconds
#ODE_str = 'lotka_volterra'  # it takes 10-20 seconds
#ODE_str = 'lorenz96' # Runtime depends on the number of states. It takes
# around 1.5 minutes for the case of d=10 in our code.


if(ODE_str == 'fitzhugh_nagumo'):
    x0 = np.array([-1, 1])
    true_param = [.2, .5, 3]
    end_t = 20
    dt = .05
    X,Y,dt = simulate(ODE_str, x0, true_param , end_t, dt, noise_var = .5)
    init_param = np.array([2, 2, 5])  # initialization of the parameters
elif(ODE_str == 'lotka_volterra'):
    x0 = np.array([5, 3])
    true_param = [2, 1, 4, 1]
    end_t = 4
    dt = .1
    X,Y,dt = simulate(ODE_str, x0, true_param, end_t, dt, noise_var = .5)
    init_param = np.array([2, 2, 5, 2])  # initialization of the parameters
elif(ODE_str == 'rossler'):
    x0 = np.array([1.13293, -1.74953, 0.02207])
    true_param = [.2, .2, 3]
    end_t = 20
    dt = .1
    X,Y,dt = simulate(ODE_str, x0, true_param, end_t, dt, noise_var = .5)
    init_param = np.array([2, 2, 5])  # initialization of the parameters
elif(ODE_str == 'lorenz96'):
    d = 10  # set the number of states
    x0 = 8 * np.ones((d,))
    x0[int(d / 2)] += .001
    true_param = [8]
    end_t = 50
    dt = 1
    # Burn in for 50 seconds
    X, Y, dt = simulate(ODE_str, x0, true_param, end_t, dt, noise_var=.5)
    #The last state after 50s is our initial state
    x0 = X[-1, :]
    end_t = 2
    dt = .01
    X, Y, dt = simulate(ODE_str, x0, true_param, end_t, dt, noise_var=.5)
    init_param = np.array([1000])


# determine the type of discretization: Euler vs three-step Adam-Bashforth(ad3)
appr = 'ad3'
# appr = euler

# learn the parameters, estimated states, and predicted states
params,est_X, pred_X = fit_direct(Y,dt,init_param,ODE_str,appr,lam=1,max_iters=20000,
                           tol=1e-8)


error = np.sum((pred_X - X)**2)
print('prediction error = ', error, 'params = ', params)

# draw the results: visualize the first dimension
xax = np.arange(X.shape[0]).reshape(-1,1)*dt
plt.scatter(xax, X[:,0],s=4,label = 'clean states',color='g')
plt.scatter(xax, Y[:,0],s=4, label = 'noisy observations',color='m')
plt.scatter(xax, pred_X[:,0],s=4, label = 'predicted states', color='b')
plt.legend(fancybox=True, framealpha=0.01)
plt.show()

