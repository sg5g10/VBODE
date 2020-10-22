import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import torch
import torch.nn.functional as F
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from inference.inference import prepare_symbolic, run_inference
from ode_systems.forward_sensitivity_solvers import ForwardSensTorchJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensTorchJacobians

### Define ODE right hand side, 
### in a way to facilitate both pure Python and torch ###  
def r(y, t, p, tch=False):   
  dS_dt = - (p[0] * y[1] * y[0]) 
  dI_dt = (p[0] * y[1] * y[0]) - (p[1] * y[1])
  dR_dt = p[1] * y[1]  
  return torch.stack([dS_dt,dI_dt,dR_dt]) if tch else [dS_dt,dI_dt,dR_dt]

### Define pyro generative model ###    
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
            sns.kdeplot(vb_params[:, i], color='orange', linewidth = 2.5, label='Variational For VJP')
            sns.kdeplot(mc_params[:, i], color='orange', linewidth = 2.5, label='Variational Adj VJP')
        else:
            sns.kdeplot(vb_params[:, i], linewidth = 2.5, color='orange')
            sns.kdeplot(mc_params[:, i], linewidth = 2.5, color='orange')
            
        if i%2==0:
            plt.ylabel('Frequency')
        plt.xlabel(param_names[i])        
        if i<1:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2, fontsize=18)
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit stochastic Lotka-Volterra model')
    parser.add_argument('--iterations', type=int, default=2000, metavar='N',
                    help='number of VI iterations') 
    parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                    help='number of draws from variational posterior ')                            
    args = parser.parse_args()

    ### Generate Tristan da Cunha Data ###       
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
            
    ### Run variational inference ###
    param_names = [r"$\beta$",r"$\gamma$", r"$s_0$"]  
    print('Using AD (Torch) for Jacobians \n')
    print('Using VJP by Forward Sensitivity')   
    sir_ode_model = ForwardSensTorchJacobians(r, 3, 5, \
        times, 1e-5, 1e-6, [0.9,0.1,0.0])
    sir_ode_model.set_unknown_y0()    

    method = 'VI'
    lr = 0.5
    vb_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr=lr, num_particles=1, return_sites=("ode_params1","ode_params2","ode_params3"))
    vb_params_for=np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                        vb_samples['ode_params2'][:,None].detach().numpy(),
                        vb_samples['ode_params3'][:,None].detach().numpy()
                        ),axis=1)


    print('Using VJP by Adjoint Sensitivity')
    sir_ode_model = AdjointSensTorchJacobians(r, 3, 5, \
        times, 1e-5, 1e-6, [0.9,0.1,0.0])
    sir_ode_model.set_unknown_y0()    
    
    vb_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr=lr, num_particles=1, return_sites=("ode_params1","ode_params2","ode_params3"))
    vb_params_adj=np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                        vb_samples['ode_params2'][:,None].detach().numpy(),
                        vb_samples['ode_params3'][:,None].detach().numpy()
                        ),axis=1)
    
    plot_marginals(vb_params_for, vb_params_adj, param_names, rows=2)
