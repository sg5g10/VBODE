from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kde

def _kde(x, y, nbins):
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    return xi, yi, zi

def pairwise(samples, parameter_names=None, title=None, saveto=None, nbins=100):

    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize":(9,11),"font.size":25,"axes.titlesize":25,"axes.labelsize":25,
           "xtick.labelsize":25, "ytick.labelsize":25},style="white")

    _, n_param = samples.shape
    fig_size = (3 * n_param, 3 * n_param)
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
  
    for i in range(n_param):
        for j in range(n_param):
            if i == j:
                axes[i, j].axis('off')

            elif i < j:
                axes[i, j].axis('off')

            else:
                x, y, z =_kde(samples[:, j], samples[:, i], nbins)
                axes[i,j].pcolormesh(x, y, z.reshape(x.shape))    
            
            if i < n_param - 1:
                axes[i, j].set_xticklabels([])
            else:
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)

            if j > 0:
                axes[i, j].set_yticklabels([])

        if parameter_names is not None:
            axes[-1, i].set_xlabel(parameter_names[i])
            
        else:
            axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        if i == 0:
            axes[i, 0].set_ylabel('Frequency')
        else:
            if parameter_names is not None:
                axes[i, 0].set_ylabel(parameter_names[i])
            else:
                axes[i, 0].set_ylabel('Parameter %d' % (i + 1))
    if title is not None:
        fig.suptitle(title,fontsize=40)
    fig.tight_layout()
    fig.subplots_adjust(top=1.05)
    
    if saveto is not None:
        plt.savefig(saveto, dpi=250)
    
    return fig, axes