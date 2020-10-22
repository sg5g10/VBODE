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
    S, dS, R, Rs, Rpp=y
    p1, p2, p3, p4, p5, p6 = p

    dS_dt = -p1*S - p2*S*R + p3*Rs
    d_dS_dt = p1*S
    dR_dt = -p2*S*R + p3*Rs + p5*(Rpp/(p6 + Rpp))
    dRs_dt = p2*S*R - p3*Rs - p4*Rs
    dRpp_dt = p4*Rs - p5*(Rpp/(p6 + Rpp))
    
    return dS_dt,d_dS_dt,dR_dt,dRs_dt,dRpp_dt

### Define generative model ###     
class ProteinGenModel(PyroModule):
    def __init__(self, ode_op, ode_model):        
        super(ProteinGenModel, self).__init__()
        self._ode_op = ode_op
        self._ode_model = ode_model
        self.ode_params = PyroSample(dist.Beta(
            torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0]),2.0).to_event(1) )      
        
    def forward(self, data): 
        
        scale = pyro.sample("scale", dist.HalfNormal(0.1))
        sd = scale.view((-1,)).unsqueeze(1)            
        param_shape = 6
        states = self._ode_op.apply(self.ode_params.view((-1,param_shape)), \
            (self._ode_model,))
        for i in range(len(data)):
            pyro.sample("obs_{}".format(i), dist.Normal(loc = states[...,i,:], \
                scale = sd).to_event(1), obs=data[i,:])           
        return states

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
            sns.kdeplot(mc_params[:, i], color='orange', linewidth = 2.5, label='MCMC')
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
        description='Fit SIR model')
    parser.add_argument('--adjoint', type=bool, default=False, metavar='N',
                    help='Method to compute VJP')
    parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                    help='number of VI iterations') 
    parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                    help='number of draws from variational posterior ')  
    parser.add_argument('--num_samples', type=int, default=500, metavar='N',
                    help='number of NUTS samples')    
    parser.add_argument('--warmup_steps', type=int, default=500, metavar='N',
                    help='number of NUTS warmup_steps')                
    args = parser.parse_args()

    ### Generate the symbolic system and 
    ### artificially corrupted data ###    
    _rhs = r
    _y, _p = sym.symbols('y:5'), sym.symbols('p:6')
    rhs_f, jac_x_f, jac_p_f = prepare_symbolic(_rhs, _y, _p)
    times = np.array([0,1,2,4,5,7,10,15,20,30,40,50,60,80,100])
    pr_ode_model = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 5, 6, \
        times, 1e-5, 1e-6, [1,0,1,0,0])
    sigma = 0.01
    real_params = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3]
    sol = pr_ode_model.solve(real_params)
    np.random.seed(121) 
    Y = sol+np.random.randn(len(times),5)*sigma    
    
    ### Run inference ###
    param_names = [r"$p_1$",r"$p_2$", r"$p_3$",r"$p_4$",r"$p_5$", r"$p_6$", r"$\sigma$"]   
    real_params.append(0.01)     
    if not(args.adjoint):
        print('Using VJP by Forward Sensitivity')

        method = 'NUTS'
        NUTS_samples = run_inference(Y, ProteinGenModel, pr_ode_model, method, \
            iterations = args.num_samples, warmup_steps = args.warmup_steps)
        mc_params=np.concatenate((NUTS_samples['ode_params'],
                        NUTS_samples['scale'][:,None]),axis=1)

        method = 'VI'
        lr = 0.5
        vb_samples = run_inference(Y, ProteinGenModel, pr_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr = lr, num_particles = 1, return_sites = ("ode_params","scale","_RETURN"))
        vb_params = \
        np.concatenate((vb_samples['ode_params'].detach().numpy().reshape((args.num_qsamples,6)), \
            vb_samples['scale'].detach().numpy().reshape((args.num_qsamples,1))),axis=1)
        
        plot_marginals(vb_params, mc_params, param_names, real_params=real_params)
    else:
        print('Using VJP by Adjoint Sensitivity')
        pr_ode_model = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 5, 6, \
            times, 1e-5, 1e-6, [1,0,1,0,0])  
        
        method = 'NUTS'
        NUTS_samples = run_inference(Y, ProteinGenModel, pr_ode_model, method, \
            iterations = args.num_samples, warmup_steps = args.warmup_steps)
        mc_params=np.concatenate((NUTS_samples['ode_params'],
                        NUTS_samples['scale'][:,None]),axis=1)

        method = 'VI'
        lr = 0.5
        vb_samples = run_inference(Y, ProteinGenModel, pr_ode_model, method, \
            iterations = args.iterations, num_samples = args.num_qsamples, \
            lr = lr, num_particles = 1, return_sites = ("ode_params","scale","_RETURN"))
        vb_params = \
        np.concatenate((vb_samples['ode_params'].detach().numpy().reshape((args.num_qsamples,6)), \
            vb_samples['scale'].detach().numpy().reshape((args.num_qsamples,1))),axis=1)   
          
        plot_marginals(vb_params, mc_params, param_names, real_params=real_params)
   
    
