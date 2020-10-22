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
    S, dS, R, Rs, Rpp=y
    p1, p2, p3, p4, p5, p6 = p

    dS_dt = -p1*S - p2*S*R + p3*Rs
    d_dS_dt = p1*S
    dR_dt = -p2*S*R + p3*Rs + p5*(Rpp/(p6 + Rpp))
    dRs_dt = p2*S*R - p3*Rs - p4*Rs
    dRpp_dt = p4*Rs - p5*(Rpp/(p6 + Rpp))
    
    return dS_dt,d_dS_dt,dR_dt,dRs_dt,dRpp_dt


def plot_marginals(vb_params, mc_params, param_names, plot_name, real_params=None, rows=4):
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
_y, _p = sym.symbols('y:5'), sym.symbols('p:6')
rhs_f, jac_x_f, jac_p_f = prepare_symbolic(_rhs, _y, _p)
times = np.array([0,1,2,4,5,7,10,15,20,30,40,50,60,80,100])

pr_ode_model_for = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 5, 6, \
    times, 1e-5, 1e-6, [1,0,1,0,0])
pr_ode_model_adj = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 5, 6, \
    times, 1e-5, 1e-6, [1,0,1,0,0])
pr_ode_model_adj.set_checkpointed()

sigma = 0.01
real_params = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3]
sol = pr_ode_model_for.solve(real_params)
np.random.seed(121)   
Y = sol+np.random.randn(len(times),5)*sigma

param_filename = './results/pr_vi_for.p'
vb_for = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/pr_vi_adj.p'
vb_adj = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/pr_hmc_for.p'
mc_for = pickle.load( open( param_filename , "rb" ) )[::2,:]
param_filename = './results/pr_hmc_adj.p'
mc_adj = pickle.load( open( param_filename , "rb" ) )[::2,:]

param_names = [r"$p_1$",r"$p_2$", r"$p_3$",r"$p_4$",r"$p_5$", r"$p_6$", r"$\sigma$"]   
real_params = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3, 0.01]
plot_marginals(vb_for, mc_for, param_names, './figures/ppc_protein/marginals_for', real_params=real_params, rows=4)
plot_marginals(vb_adj, mc_adj, param_names, './figures/ppc_protein/marginals_adj', real_params=real_params, rows=4)

pairwise(vb_for, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_vb_for.png', nbins=100)
pairwise(vb_adj, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_vb_adj.png', nbins=100)
pairwise(mc_for, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_mc_for.png', nbins=100)
pairwise(mc_adj, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_mc_adj.png', nbins=100)


mc_for_ppc = []
vb_for_ppc = []
mc_adj_ppc = []
vb_adj_ppc = []
PPC_samples = 1000
for i in range(PPC_samples):

    mc_for_ppc.append(pr_ode_model_for.solve(mc_for[i,:-1]))
    vb_for_ppc.append(pr_ode_model_for.solve(vb_for[i,:-1]))

    mc_adj_ppc.append(pr_ode_model_adj.solve(mc_adj[i,:-1]))
    vb_adj_ppc.append(pr_ode_model_adj.solve(vb_adj[i,:-1]))

mc_for_ppc = np.array(mc_for_ppc)
vb_for_ppc = np.array(vb_for_ppc)
mc_adj_ppc = np.array(mc_adj_ppc)
vb_adj_ppc = np.array(vb_adj_ppc)
for i in range(15):
    for j in range(5):      
        mc_for_ppc[:,i,j] = np.random.normal(mc_for_ppc[:,i,j],mc_for[:PPC_samples,-1])
        vb_for_ppc[:,i,j] = np.random.normal(vb_for_ppc[:,i,j],vb_for[:PPC_samples,-1])
        mc_adj_ppc[:,i,j] = np.random.normal(mc_adj_ppc[:,i,j],mc_adj[:PPC_samples,-1])
        vb_adj_ppc[:,i,j] = np.random.normal(vb_adj_ppc[:,i,j],vb_adj[:PPC_samples,-1])

mean_ppc_vb_for = vb_for_ppc.mean(axis=0)
CriL_ppc_vb_for = np.percentile(vb_for_ppc,q=.5,axis=0)
CriU_ppc_vb_for = np.percentile(vb_for_ppc,q=97.5,axis=0)

mean_ppc_mc_for = mc_for_ppc.mean(axis=0)
CriL_ppc_mc_for = np.percentile(mc_for_ppc,q=.5,axis=0)
CriU_ppc_mc_for = np.percentile(mc_for_ppc,q=97.5,axis=0)

mean_ppc_vb_adj = vb_adj_ppc.mean(axis=0)
CriL_ppc_vb_adj = np.percentile(vb_adj_ppc,q=.5,axis=0)
CriU_ppc_vb_adj = np.percentile(vb_adj_ppc,q=97.5,axis=0)

mean_ppc_mc_adj = mc_for_ppc.mean(axis=0)
CriL_ppc_mc_adj = np.percentile(mc_adj_ppc,q=.5,axis=0)
CriU_ppc_mc_adj = np.percentile(mc_adj_ppc,q=97.5,axis=0)

sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(5,7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
for i in range(5):
    plt.figure(figsize=(7, 5))
    
    plt.plot(times,mean_ppc_vb_for[:,i], color='magenta', lw=4, label='VI')
    plt.plot(times,CriL_ppc_vb_for[:,i], '--', color='magenta', lw=3)
    plt.plot(times,CriU_ppc_vb_for[:,i], '--',  color='magenta', lw=3)
    plt.plot(times,mean_ppc_mc_for[:,i], color='orange', lw=4, label='NUTS')
    plt.plot(times,CriL_ppc_mc_for[:,i], '--', color='orange', lw=3)
    plt.plot(times,CriU_ppc_mc_for[:,i], '--',  color='orange', lw=3)
    plt.plot(times,Y[:,i],'o', color='k', lw=4, ms=10.5, label='Observations')
    plt.xlim([0,100])
    plt.xlabel('Time', fontsize=0)
    plt.ylabel('Concentration', fontsize=0)
    if i==0:
        plt.legend(fontsize=15)
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.savefig('./figures/ppc_protein/pr_ppc'+str(i+1)+'.eps')
    plt.close()

    plt.figure(figsize=(7, 5))
    
    plt.plot(times,mean_ppc_vb_adj[:,i], color='magenta', lw=4, label='VI')
    plt.plot(times,CriL_ppc_vb_adj[:,i], '--', color='magenta', lw=3)
    plt.plot(times,CriU_ppc_vb_adj[:,i], '--',  color='magenta', lw=3)
    plt.plot(times,mean_ppc_mc_adj[:,i], color='orange', lw=4, label='NUTS')
    plt.plot(times,CriL_ppc_mc_adj[:,i], '--', color='orange', lw=3)
    plt.plot(times,CriU_ppc_mc_adj[:,i], '--',  color='orange', lw=3)
    plt.plot(times,Y[:,i],'o', color='k', lw=4, ms=10.5, label='Observations')
    plt.xlim([0,100])
    plt.xlabel('Time', fontsize=0)
    plt.ylabel('Concentration', fontsize=0)
    if i==0:
        plt.legend(fontsize=15)
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.savefig('./figures/ppc_protein/pr_ppc_adj'+str(i+1)+'.eps')
    plt.close()