import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from inference.inference import prepare_symbolic
from util.plot_pairwise import pairwise
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians


def r(y, t, p):
    S, I, R=y
    beta, gamma = p
    dS_dt = - (beta * I * S)
    dI_dt = (beta * I * S) - (gamma * I)
    dR_dt = gamma * I
    return dS_dt,dI_dt,dR_dt

def plot_marginals(vb_params, mc_params, param_names, plot_name, real_params=None, rows=2):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize":(9,9),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")

    for i, p in enumerate(param_names):        
        plt.subplot(rows, 2, i+1)
        if real_params is not None:
            plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i==0:
            sns.kdeplot(vb_params[:, i], color='magenta', linewidth = 2.5, label='VI')
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
    plt.savefig(plot_name+'.eps')
    plt.close()

_rhs = r
_y, _p = sym.symbols('y:3'), sym.symbols('p:2')
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

param_filename = './results/sir_vb_for.p'
vb_for = pickle.load( open( param_filename , "rb" ) )

param_filename = './results/sir_vb_adj.p'
vb_adj = pickle.load( open( param_filename , "rb" ) )

param_filename = './results/sir_hmc_for.p'
mc_for = pickle.load( open( param_filename , "rb" ) )[::2,:]

param_filename = './results/sir_hmc_adj.p'
mc_adj = pickle.load( open( param_filename , "rb" ) )[::2,:]


param_names = [r"$\beta$",r"$\gamma$", r"$s_0$"]
plot_marginals(vb_for, mc_for, param_names, './figures/ppc_sir/sir_marginals_for')
plot_marginals(vb_adj, mc_adj, param_names, './figures/ppc_sir/sir_marginals_adj')


pairwise(vb_for, parameter_names=param_names, saveto='./figures/ppc_sir/sir_pairwise_vb_for.png', nbins=100)
#pairwise(vb_adj, parameter_names=param_names, saveto='./figures/ppc_sir/sir_pairwise_vb_adj.png', nbins=100)
pairwise(mc_for, parameter_names=param_names, saveto='./figures/ppc_sir/sir_pairwise_mc_for.png', nbins=100)
#pairwise(mc_adj, parameter_names=param_names, saveto='./figures/ppc_sir/sir_pairwise_mc_adj.png', nbins=100)


sir_ode_model_for = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 3, 5, \
        times, 1e-5, 1e-6, [0.9,0.1,0.0])
sir_ode_model_adj = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 3, 5, \
        times, 1e-5, 1e-6, [0.9,0.1,0.0])
sir_ode_model_adj.set_checkpointed()
mc_for_ppc = []
vb_for_ppc = []
mc_adj_ppc = []
vb_adj_ppc = []
for i in range(1000):
    sir_ode_model_for.set_y0([mc_for[i,2],1-mc_for[i,2],0])
    mc_for_ppc.append(sir_ode_model_for.solve(mc_for[i,:2])[:,1]*300)
    sir_ode_model_for.set_y0([vb_for[i,2],1-vb_for[i,2],0])
    vb_for_ppc.append(sir_ode_model_for.solve(vb_for[i,:2])[:,1]*300)

    sir_ode_model_adj.set_y0([mc_adj[i,2],1-mc_adj[i,2],0])
    mc_adj_ppc.append(sir_ode_model_adj.solve(mc_adj[i,:2])[:,1]*300)
    sir_ode_model_adj.set_y0([vb_adj[i,2],1-vb_adj[i,2],0])
    vb_adj_ppc.append(sir_ode_model_adj.solve(vb_adj[i,:2])[:,1]*300)    

mc_for_ppc = np.random.poisson(np.array(mc_for_ppc))
vb_for_ppc = np.random.poisson(np.array(vb_for_ppc))
mc_adj_ppc = np.random.poisson(np.array(mc_adj_ppc))
vb_adj_ppc = np.random.poisson(np.array(vb_adj_ppc))

mean_ppc_vb_for = vb_for_ppc.mean(axis=0)
CriL_ppc_vb_for = np.percentile(vb_for_ppc,q=2.5,axis=0)
CriU_ppc_vb_for = np.percentile(vb_for_ppc,q=97.5,axis=0)

mean_ppc_mc_for = mc_for_ppc.mean(axis=0)
CriL_ppc_mc_for = np.percentile(mc_for_ppc,q=2.5,axis=0)
CriU_ppc_mc_for = np.percentile(mc_for_ppc,q=97.5,axis=0)

mean_ppc_vb_adj = vb_adj_ppc.mean(axis=0)
CriL_ppc_vb_adj = np.percentile(vb_adj_ppc,q=2.5,axis=0)
CriU_ppc_vb_adj = np.percentile(vb_adj_ppc,q=97.5,axis=0)

mean_ppc_mc_adj = mc_for_ppc.mean(axis=0)
CriL_ppc_mc_adj = np.percentile(mc_adj_ppc,q=2.5,axis=0)
CriU_ppc_mc_adj = np.percentile(mc_adj_ppc,q=97.5,axis=0)
times +=1
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(9,11),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
plt.figure(figsize=(15, 10))
plt.subplot(2,1,1)
plt.plot(times,Y,'o', color='k', lw=4, ms=10.5, label='Observations')
plt.plot(times,mean_ppc_vb_for, color='magenta', lw=4, label='VI')
plt.plot(times,CriL_ppc_vb_for, '--', color='magenta', lw=4)
plt.plot(times,CriU_ppc_vb_for, '--',  color='magenta', lw=4)
plt.plot(times,mean_ppc_mc_for, color='orange', lw=4, label='NUTS')
plt.plot(times,CriL_ppc_mc_for, '--', color='orange', lw=4)
plt.plot(times,CriU_ppc_mc_for, '--',  color='orange', lw=4)
plt.xlim([1,21])
plt.ylabel('Incidence', fontsize=25)
plt.xticks(times,rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.subplot(2,1,2)
plt.plot(times,Y,'o', color='k', lw=4, ms=10.5)
plt.plot(times,mean_ppc_vb_adj, color='magenta', lw=4)
plt.plot(times,CriL_ppc_vb_adj, '--', color='magenta', lw=4)
plt.plot(times,CriU_ppc_vb_adj, '--',  color='magenta', lw=4)
plt.plot(times,mean_ppc_mc_adj, color='orange', lw=4)
plt.plot(times,CriL_ppc_mc_adj, '--', color='orange', lw=4)
plt.plot(times,CriU_ppc_mc_adj, '--',  color='orange', lw=4)
plt.xlim([1,21])
plt.xlabel('Days', fontsize=25)
plt.ylabel('Incidence', fontsize=25)
plt.xticks(times,rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.subplots_adjust(hspace=0.7)
plt.tight_layout()
plt.savefig('./figures/ppc_sir/sir_ppc.png')