import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from inference.inference import prepare_symbolic, run_inference
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians

### Define ODE right hand side ###  
def r(y, t, p):
    S, I, R=y
    beta, gamma, S0, I0, R0 = p
    dS_dt = - (beta * I * S)
    dI_dt = (beta * I * S) - (gamma * I)
    dR_dt = gamma * I
    return dS_dt,dI_dt,dR_dt

### Define generative model ###    
class SIRGenModel(PyroModule):
    def __init__(self, ode_op, ode_model):        
        super(SIRGenModel, self).__init__()           
        self._ode_op = ode_op
        self._ode_model = ode_model

        self.ode_params1 = PyroSample(dist.Gamma(2,1))
        self.ode_params2 = PyroSample(dist.Gamma(2,1))
        self.ode_params3 = PyroSample(dist.Beta(0.5,0.5)) 
        
    def forward(self, data): 
        N_pop = 300
                
        p1 = self.ode_params1.view((-1,))
        p2 = self.ode_params2.view((-1,))
        p3 = self.ode_params3.view((-1,))
        R0 = pyro.deterministic('R0', torch.zeros_like(p1))
        ode_params = torch.stack([p1,p2,p3,1- p3,R0], dim=1)
        SIR_sim = self._ode_op.apply(ode_params, (self._ode_model,))
        
        for i in range(len(data)):
            pyro.sample("obs_{}".format(i), dist.Poisson(SIR_sim[...,i,1]*N_pop), obs=data[i])
        return SIR_sim

def plot_marginals(vb_params, mc_params, param_names, real_params=None, rows=4):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize":(9,9),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")

    for i, p in enumerate(param_names):        
        plt.subplot(rows, 2, i+1)
        if real_params is not None:
            plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i==0:
            sns.kdeplot(vb_params[:, i], color='magenta', linewidth = 2.5, label='Variational')
            sns.kdeplot(mc_params[:, i], color='orange', linewidth = 2.5, label='NUTS')
        else:
            sns.kdeplot(vb_params[:, i], linewidth = 2.5, color='magenta')
            sns.kdeplot(mc_params[:, i], linewidth = 2.5, color='orange')
            
        if i%2==0:
            plt.ylabel('Frequency')
        plt.xlabel(param_names[i])        
        if i<1:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2, fontsize=18)
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit Protein Transduction model')
    parser.add_argument('--adjoint', type=bool, default=False, metavar='N',
                    help='Method to compute VJP')
    parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                    help='number of VI iterations') 
    parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                    help='number of draws from variational posterior ')  
    parser.add_argument('--num_samples', type=int, default=1000, metavar='N',
                    help='number of NUTS post warm-up samples')    
    parser.add_argument('--warmup_steps', type=int, default=500, metavar='N',
                    help='number of NUTS warmup_steps')                
    args = parser.parse_args()

    ### Generate the symbolic system and Tristan da Cunha Data ###       
    _rhs = r
    _y, _p = sym.symbols('y:3'), sym.symbols('p:5')
    rhs_f, jac_x_f, jac_p_f = prepare_symbolic(_rhs, _y, _p)
    times = np.arange(0,21,1)
    
    data =np.array([
            [1, 0],     # day 1
            [1, 0],
            [3, 0],
            [7, 0],
            [6, 5],     # day 5
            [10, 7],
            [13, 8],
            [13, 13],
            [14, 13],
            [14, 16],    # day 10
            [17, 17],
            [10, 24],
            [6, 30],
            [6, 31],
            [4, 33],    # day 15
            [3, 34],
            [1, 36],
            [1, 36],
            [1, 36],
            [1, 36],    # day 20
            [0, 37],    # day 21
        ])
    I, R = data[:,0], data[:,1]
    Y = I
            
    ### Run inference ###
    param_names = [r"$\beta$",r"$\gamma$", r"$s_0$"]
    if not(args.adjoint):
        print('Using VJP by Forward Sensitivity')   
        sir_ode_model = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 3, 5, \
            times, 1e-5, 1e-6, [0.9,0.1,0.0])
        sir_ode_model.set_unknown_y0()    
        method = 'NUTS'
        NUTS_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.num_samples, warmup_steps = args.warmup_steps)
        mc_params=np.concatenate((NUTS_samples['ode_params1'][:,None],
                            NUTS_samples['ode_params2'][:,None],
                            NUTS_samples['ode_params3'][:,None]
                            ),axis=1)    
                                  
        method = 'VI'
        lr = 0.5
        vb_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr=lr, num_particles=1, return_sites=("ode_params1","ode_params2","ode_params3"))
        vb_params=np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                            vb_samples['ode_params2'][:,None].detach().numpy(),
                            vb_samples['ode_params3'][:,None].detach().numpy()
                            ),axis=1)

        plot_marginals(vb_params, mc_params, param_names, rows=2)
    else:
        print('Using VJP by Adjoint Sensitivity')
        sir_ode_model = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 3, 5, \
            times, 1e-5, 1e-6, [0.9,0.1,0.0])
        sir_ode_model.set_unknown_y0()    
        method = 'NUTS'
        NUTS_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.num_samples, warmup_steps = args.warmup_steps)
        mc_params=np.concatenate((NUTS_samples['ode_params1'][:,None],
                            NUTS_samples['ode_params2'][:,None],
                            NUTS_samples['ode_params3'][:,None]
                            ),axis=1)    
        
        method = 'VI'
        lr = 0.5
        vb_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr=lr, num_particles=1, return_sites=("ode_params1","ode_params2","ode_params3"))
        vb_params=np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                            vb_samples['ode_params2'][:,None].detach().numpy(),
                            vb_samples['ode_params3'][:,None].detach().numpy()
                            ),axis=1)
        plot_marginals(vb_params, mc_params, param_names, rows=2)
        
