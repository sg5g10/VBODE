import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from inference.inference import prepare_symbolic
from CPP import lvssa

def gen_ssa(times, c):
    x_prev = np.array([100,100])
    ssa = []
    c[1] = c[1]/100
    c = np.array(c)
    for i in range(1,len(times)):
        x_next = np.array(lvssa.LV(c, x_prev, times[i-1], times[i]))
        ssa.append(x_next)
        x_prev = x_next
    ss = np.array(ssa)
    x0 = np.array([100,100]).reshape((1,2))
    ss = np.concatenate((x0, ss),axis=0)
    return ss

def plot_marginals(vb_params, mc_params, param_names, plot_name, real_params=None, rows=2):
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
    plt.savefig(plot_name+'.eps')
    plt.close()


times2 = np.arange(0,50,5)
times = np.arange(0,50,0.1)
Y = np.array([[100, 100],
 [343, 213],
 [ 28, 354],
 [ 41, 137],
 [255, 101],
 [ 71, 416],
 [ 29, 136],
 [186,  76],
 [141, 449],
 [ 27, 189]]
)

param_names = [r"$c_1$",r"$c_2 \times 100$", r"$c_3$"]   
real_params = np.array([0.53, 0.25, 0.3])

param_filename = './results/lna_abc_ssa.p'
abc = pickle.load( open( param_filename , "rb" ) )

param_filename = './results/lna_vb_for.p'
vb_for = pickle.load( open( param_filename , "rb" ) )

"""
plot_marginals(vb_for, abc, param_names, './figures/ppc_protein/marginals_for', real_params=real_params, rows=4)
plot_marginals(vb_for, abc, param_names, './figures/ppc_protein/marginals_adj', real_params=real_params, rows=4)

pairwise(vb_for, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_vb_for.png', nbins=100)
pairwise(vb_for, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_vb_for.png', nbins=100)
pairwise(mc_for, parameter_names=param_names, saveto='./figures/ppc_protein/pairwise_mc_for.png', nbins=100)
"""

abc_ppc = []
vb_for_ppc = []
vb_adj_ppc = []
PPC_samples = 100
for i in range(PPC_samples):
    abc_ppc.append(gen_ssa(times,abc[i,:]))
    vb_for_ppc.append(gen_ssa(times,vb_for[i,:]))    

abc_ppc = np.array(abc_ppc)
vb_for_ppc = np.array(vb_for_ppc)


mean_ppc_abc = abc_ppc.mean(axis=0)
CriL_ppc_abc = np.percentile(abc_ppc,q=2.5,axis=0)
CriU_ppc_abc = np.percentile(abc_ppc,q=97.5,axis=0)

mean_ppc_vb_for = vb_for_ppc.mean(axis=0)
CriL_ppc_vb_for = np.percentile(vb_for_ppc,q=2.5,axis=0)
CriU_ppc_vb_for = np.percentile(vb_for_ppc,q=97.5,axis=0)



sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(9,11),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
plt.figure(figsize=(15, 10))
plt.subplot(2,1,1)
plt.plot(times2,Y[:,0],'o', color='k', lw=4, ms=10.5, label='Observations')
plt.plot(times,mean_ppc_abc[:,0], color='magenta', lw=4, label='ABC-SMC')
plt.plot(times,CriL_ppc_abc[:,0], color='magenta', lw=3)
plt.plot(times,CriU_ppc_abc[:,0], color='magenta', lw=3)
plt.plot(times,mean_ppc_vb_for[:,0], color='orange', lw=4, label='VI-FOR')
plt.plot(times,CriL_ppc_vb_for[:,0], color='orange', lw=3)
plt.plot(times,CriU_ppc_vb_for[:,0], color='orange', lw=3)
plt.xlim([0,50])
plt.ylabel('Prey', fontsize=25)
plt.xticks(times2+5,rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.subplot(2,1,2)
plt.plot(times2,Y[:,1],'o', color='k', lw=4, ms=10.5)
plt.plot(times,mean_ppc_abc[:,1], color='magenta', lw=4)
plt.plot(times,CriL_ppc_abc[:,1], color='magenta', lw=3)
plt.plot(times,CriU_ppc_abc[:,1], color='magenta', lw=3)
plt.plot(times,mean_ppc_vb_for[:,1], color='orange', lw=4)
plt.plot(times,CriL_ppc_vb_for[:,1], color='orange', lw=3)
plt.plot(times,CriU_ppc_vb_for[:,1], color='orange', lw=3)

plt.xlim([0,50])
plt.xlabel('Time', fontsize=25)
plt.ylabel('Predator', fontsize=25)
plt.xticks(times2+5,rotation=45, fontsize=25)
plt.yticks(fontsize=25)
plt.subplots_adjust(hspace=0.7)
plt.tight_layout()
plt.savefig('./figures/ppc_lna/lna_ppc.png')