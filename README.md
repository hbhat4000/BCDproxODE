<h3>Python code implementing the BCDprox algorithm for ODE parameter estimation and filtering</h3>

<b>Reference:</b>

A Block Coordinate Descent Proximal Method for Simultaneous Filtering and Parameter Estimation,
Ramin Raziperchikolaei and Harish S. Bhat,
Proceedings of the 36th International Conference on Machine Learning,
ICML 2019, Long Beach, CA, June 10-15, 2019.

<b>Abstract:</b>
    
We propose and analyze a block coordinate descent proximal algorithm (BCD-prox) for simultaneous filtering and parameter estimation of ODE models.  As we show on ODE systems with up to d=40 dimensions, as compared to state-of-the-art methods, BCD-prox exhibits increased robustness (to noise, parameter initialization, and hyperparameters), decreased training times, and improved accuracy of both filtered states and estimated parameters.  We show how BCD-prox can be used with multistep numerical discretizations, and we establish convergence of BCD-prox under hypotheses that include real systems of interest.    

<h4>List of files:</h4>

- demo.py: This is the main file. You need to run this file for the parameter 
and state estimation. You can select the type of ODE, amount of the noise, true 
parameters, etc., before running the algorithm.

- AUX/fit_direct.py: The file contains the main loop of our algorithm, which 
optimizes X and theta alternately.

- AUX/objectives.py: This contains two functions. 1) X_obj() is the objective 
function defined over the states given the parameters (eq.(8) [Euler] or eq. 
(13b) [multistep] of the paper). 2) param_obj() is the objective function over 
the parameters given the states (eq.(7) [Euler] or eq. (13a) [multistep] of the 
paper).

- AUX/simulate.py: This file contains two functions. 1) simulate(), which 
creates clean states and noisy observations for the ODEs. 2) predict(), which 
returns the predicted states given the initialization and the estimated 
parameters.

- ODEs/lotka_volterra.py: contains the functions for the ODE of the 
Lotka_Volterra model (eq.(16) of the paper).

- ODEs/fitzhugh_nagumo.py: contains the functions for the ODE of the 
Fitzhugh_Nagumo model (eq.(17) of the paper).

- ODEs/rossler.py: contains the functions for the ODE of the Rossler model 
(eq.(18) of the paper).

- ODEs/lorenz96.py: contains the functions for the ODE of the Lorenz96 model 
(eq.(19) of the paper).


<h4>How to add a new ODE:</h4>

Let's assume that the name of this ODE is "new". You need to do the
following:

1. Add a new file "new.py" to the ODEs/ folder. This file contains two
functions that gets the state and returns the derivatives (eq. (1) of the 
paper). The name of the functions have to be "new_ode()" and "new_ode_vec()". 
You can follow what we did for the four ODEs inside the ODEs/ folder.

2. You need to modify the demo.py file. Define a new ODE_str sting with the 
name "new", set the parameters (dt, end_time, true parameters, noise,
etc.).

<h4>Installation:</h4>

We ran the demo file on Linux with the PyCharm IDE. We used the Conda 
environment. Here are the steps:

1. Install Conda in Linux: 
    - https://conda.io/docs/user-guide/getting-started.html#starting-conda

2. create an environment with anaconda packages ( we call it "direct"):
    - conda create -n direct python=3.6 anaconda

3. Install autograd in the direct environment:
    - source activate direct
    - conda install -c omnia autograd 
    For more information about autograd see:
    https://github.com/HIPS/autograd

4. Open our folder in PyCharm and set the direct environment in the PyCharm:
    - File -> Settings -> Project -> Project Interpreter -> click on the 
    setting gear at the top -> click on the add -> click on the VirtualEnv 
Environment on the left panel -> click on the existing environment -> select 
the address of the python inside your conda environment. In my linux 
system, the address is: ~/.conda/envs/direct/bin/python
    
5. Run the demo file!


    

