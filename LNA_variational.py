import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
import pyro.distributions as dist
import torch.distributions as D
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from inference.inference import prepare_symbolic, run_inference
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians

### Define the LNA coupled ODE system ###
def r(y, t, p):
  z = y[:2]
  V = y[2:]
  V = np.reshape(V,(2,2))
  c = p
  z = z
  S = np.array([[1,-1,0],[0,1,-1]])
  F = np.array([[c[0], 0],[c[1]*z[1], c[1]*z[0]],[0, c[2]]])
  H = np.array([c[0]*z[0], c[1]*z[0]*z[1], c[2]*z[1]])
  A = np.diag([c[0]*z[0], c[1]*z[0]*z[1], c[2]*z[1]])

  dz_dt = np.matmul(S,H)
  dV_dt = np.matmul(V, np.matmul(F.T, S.T)) + (S.dot(A).dot(S.T)) + np.matmul(S, np.matmul(F, V))  
  return np.hstack((dz_dt, np.ravel(dV_dt)))

### Define generative model ### 
class LNAGenModel(PyroModule):
    def __init__(self, ode_op, ode_model):        
        super(LNAGenModel, self).__init__()
        self._ode_op = ode_op
        self._ode_model = ode_model
        self.ode_params1 = PyroSample(dist.Beta(2,1))
        self.ode_params2 = PyroSample(dist.HalfNormal(1)) 
        self.ode_params3 = PyroSample(dist.Beta(1,2))      
        
    def forward(self, data): 
        p1 = self.ode_params1.view((-1,))
        p2 = self.ode_params2.view((-1,))
        p3 = self.ode_params3.view((-1,))

        for i in range(1, len(data)):
            z_start = torch.stack([*data[i-1,:],*torch.zeros(4)])
            z_cov = self._ode_op.apply(torch.stack([p1,p2/100,p3], dim=-1), (self._ode_model, z_start))[...,-1,:]
            
            pyro.sample("obs_{}".format(i), D.MultivariateNormal(z_cov[...,:2], covariance_matrix=z_cov[...,2:].view((-1,2,2))), obs=data[i,:])
            
        return z_cov 

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
        description='Fit stochastic Lotka-Volterra model')
    parser.add_argument('--iterations', type=int, default=2000, metavar='N',
                    help='number of VI iterations') 
    parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                    help='number of draws from variational posterior ')                         
    args = parser.parse_args()
    print('Iters: \n',args.iterations)
    print('q_samples: \n',args.num_qsamples)
    ### Generate the symbolic system and define the data generated earlier ### 
    _rhs = r
    _y, _p = sym.symbols('y:6'), sym.symbols('p:3')
    rhs_f, jac_x_f, jac_p_f = prepare_symbolic(_rhs, _y, _p)
    times = np.array([0,5])
    
    Y = np.array([[100, 100], [343, 213], [ 28, 354], [ 41, 137], [255, 101], \
        [ 71, 416], [ 29, 136], [186,  76], [141, 449], [ 27, 189]])


    ### Run inference ###
    param_names = [r"$c_1$",r"$c_2 \times 100$", r"$c_3$"]   
    real_params = np.array([0.53, 0.25, 0.3])
    
    print('Using VJP by Forward Sensitivity')      
    lna_ode_model = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 6, 3, \
        times, 1e-5, 1e-6, [100, 100, 0, 0 ,0 ,0])        
    method = 'VI'
    lr = 0.5
    vb_samples = run_inference(Y, LNAGenModel, lna_ode_model, method, iterations=args.iterations, \
        lr = lr, num_particles = 1, num_samples = args.num_qsamples, \
            return_sites = ("ode_params1","ode_params2","ode_params3"))
    vb_params_for = np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                        vb_samples['ode_params2'][:,None].detach().numpy(),
                        vb_samples['ode_params3'][:,None].detach().numpy()
                        ),axis=1)

    
    print('Using VJP by Adjoint Sensitivity')
    lna_ode_model = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 6, 3, \
    times, 1e-5, 1e-6, [100, 100, 0, 0 ,0 ,0])        

    vb_samples = run_inference(Y, LNAGenModel, lna_ode_model, method, iterations=args.iterations, \
        lr = lr, num_particles = 1, num_samples = args.num_qsamples, \
            return_sites = ("ode_params1","ode_params2","ode_params3"))
    vb_params_adj = np.concatenate((vb_samples['ode_params1'][:,None].detach().numpy(),
                        vb_samples['ode_params2'][:,None].detach().numpy(),
                        vb_samples['ode_params3'][:,None].detach().numpy()
                        ),axis=1)

    plot_marginals(vb_params_for, vb_params_adj, param_names, real_params = real_params, rows=2)