import numpy as np
import os
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm
import colorcet as cc
from matplotlib import rcParams

config = {
    # 'figure.figsize': (8, 6),
    'lines.linewidth': 2.0, 
    "font.family": 'Arial',
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    "axes.unicode_minus": False,
    'axes.linewidth': 2.0
}
rcParams.update(config)


from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))
color_dict.update({'CornflowerBlue': '#6495ED', 
                   'DarkRed': '#8B0000',
                   'IndianRed':'#CD5C5C',
                   })
color_labels = ['CornflowerBlue', 'DarkRed', 'IndianRed', 'brown', 'darkcyan', 'purple']

initPDFs = ['uniform', 'beta', 'multimodal'] 
initColors = {
              'uniform': 'dodgerblue', 
              'beta': 'tomato', 
              'multimodal': 'purple',
              'mix_gaussian_2d': 'purple',
              'mix_gaussian_2d': 'crimson',
              'olympic_mix': 'purple',
             }

FIGTYPE = 'pdf'

# linestyles = ['-','--','-.',':','-','--','-.',':']
# linemarkers = ["s","d", "o",">","*","x","<",">"]
# linemarkerswidth = [3,2,2,2,4,2,2,2]

# FONTSIZE=18
# font = {'size':FONTSIZE, 'family':'Times New Roman'}
# matplotlib.rc('xtick', labelsize=FONTSIZE) 
# matplotlib.rc('ytick', labelsize=FONTSIZE) 
# matplotlib.rc('font', **font)
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
# matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# rc('text', usetex=True)
# plt.rcParams["text.usetex"] = True
# plt.rcParams['xtick.major.pad']='10'
# plt.rcParams['ytick.major.pad']='10'


def projection2D(data, color='dodgerblue', figname='/figures/projection2D.pdf'):
    assert data.ndim==2
    traj_len, dim = data.shape[0], data.shape[1]
    fig = plt.figure(figsize=(7, 6))
    plt.plot(data[:,0], data[:,1], color=color)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.savefig(os.getcwd()+figname, bbox_inches="tight", dpi=600)



def plotPDFs_1d(times_hist, scalarDDE_hists, initPDF='multimodal', initColor='purple', figname='/figures/chaoticDDE_n1_PDFs.pdf'):
    ncols = 5
    fig, axes = plt.subplots(nrows=1, ncols=ncols+1, figsize=(28, 5))
    for j in range(ncols+1):
        col = "t"+str(j)
        pdf = [float(x) for x in np.array(scalarDDE_hists[[col]][1:])]
        pdf = np.array(pdf)
        if j==ncols:
            statistic, p_value = stats.normaltest(pdf)
        plt.subplot(1, ncols+1, j+1)
        plt.title("$t={:.1f}$".format(times_hist[j][0]))
        sns.kdeplot(pdf, color=initColor, label=initPDF)
        plt.xlabel(r"$x$")
        if j==0:
            plt.ylabel(r"$\rho(x)$")
        else:
            plt.ylabel("")
        if j==ncols:
            statistic, p_value = stats.normaltest(pdf)
            xmin, xmax = plt.gca().get_xlim()
            ymin, ymax = plt.gca().get_ylim()
            plt.text(0.9*xmax, 0.9*ymax, 'p={:.2f}'.format(p_value), 
                     fontdict={'family': 'serif', 'size': 16, 'color': 'black'}, ha='right', va='top')
        axes[j].legend([], frameon=False)
    plt.tight_layout()

    plt.savefig(os.getcwd()+figname, bbox_inches="tight", dpi=600)
    # plt.savefig(os.getcwd()+"/figures/chaoticDDE_n1_{:}_PDFs.pdf".format(initPDF), bbox_inches="tight", dpi=600)

def plotMSD_1d(scalarDDE_msd, initPDF='multimodal', initColor='purple', figname='/figures/chaoticDDE_n1_MSDs.pdf'):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
    plt.plot(scalarDDE_msd[:,0], scalarDDE_msd[:,1], color=initColor, label=initPDF)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\langle (x(t)-x(0))^2\rangle$")
    plt.tight_layout()
    axes.grid(color='gray', linestyle=':', linewidth=1, alpha=0.9)
    plt.savefig(os.getcwd()+figname, bbox_inches="tight", dpi=600)
    # plt.savefig(os.getcwd()+"/figures/chaoticDDE_n1_{:}_MSDs.pdf".format(initPDF), bbox_inches="tight", dpi=600)


def plotPDFs_2d(times_hist, DDE_hists, initPDF, figname):
    n = 2
    ncols = 5
    fig, axes = plt.subplots(nrows=1, ncols=ncols+1, figsize=(24, 4))
    for j in range(ncols+1):
        samples = DDE_hists[:,j*n:j*n+n]  # (n_init, 2)
        xmax, ymax = np.max(samples, axis=0)
        xmin, ymin = np.min(samples, axis=0)
        plt.subplot(1, ncols+1, j+1) 
        plt.scatter(samples[:,0], samples[:,1], s=0.4, alpha=1.0)
        plt.title("$t={:.1f}$".format(times_hist[j][0]))
        plt.tight_layout()
        plt.grid(color='gray', linestyle=':', linewidth=1, alpha=0.9)
        x_adjustment = (xmax - xmin) * 0.15
        y_adjustment = (ymax - ymin) * 0.15
        plt.axis([xmin-x_adjustment, xmax+x_adjustment, ymin-y_adjustment, ymax+y_adjustment])
        plt.gca().axis('equal')

    plt.savefig(os.getcwd()+figname, bbox_inches="tight", dpi=600)


def plotMSD_2d(DDE_msd, initPDF, color, figname):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    plt.plot(DDE_msd[:,0], DDE_msd[:,1], linestyle='-', color=color, label=initPDF)
    plt.plot(DDE_msd[:,0], DDE_msd[:,2], linestyle=':', color=color)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\langle (x(t)-x(0))^2\rangle$")
    plt.tight_layout()
    axes.grid(color='gray', linestyle=':', linewidth=1, alpha=0.9)
    plt.savefig(os.getcwd()+figname, bbox_inches="tight", dpi=600)


def PlotTrainingLosses(model, loss_train, loss_val, min_val_error):
    
    min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
    fig_path = model.getFigureDir() + "/{:}_loss_total.{:}".format(model.rnnChaosName, FIGTYPE) 
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title("Validation error {:.10f}".format(min_val_error))
    plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
    plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
    plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"Loss")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    loss_train = np.array(loss_train)
    loss_val = np.array(loss_val)
    if (np.all(loss_train[~np.isnan(loss_train)]>0.0) and np.all(loss_val[~np.isnan(loss_val)]>0.0)):
        fig_path = model.getFigureDir() + "/{:}_loss_total_log.{:}".format(model.rnnChaosName, FIGTYPE) 
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), np.log10(loss_train), color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), np.log10(loss_val), color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, np.log10(min_val_error), "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Log${}_{10}$(Loss)")
        plt.legend(loc="upper left", frameon=False)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()


def PlotPredictions(model, mode, target, prediction, dt, lambda_max=None, plotEmbed=False):
    if lambda_max is None:
        tpts = dt * np.arange(np.shape(prediction)[0])
    else:
        tpts = dt * np.arange(np.shape(prediction)[0]) * lambda_max

    if plotEmbed:
        # plot the embeded trajectory
        fig_path = model.getFigureDir() + "/{:}_{:}_x12.{:}".format(model.rnnChaosName, mode, FIGTYPE)
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot()
        plt.plot(tpts, prediction[:,0], color=color_dict['deepskyblue'], linewidth=2.0, label='prediction')
        plt.plot(tpts, target[:,0], color=color_dict['tomato'], linewidth=2.0, label='target')
        ax.legend(bbox_to_anchor=(0.25, 1.2), loc="upper left", ncol=2, frameon=False)
        ax.set_xlabel(r"$\Lambda_1 t$")
        ax.set_ylabel(r"$x$")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    else:
        # plot the first three components
        fig_path = model.getFigureDir() + "/{:}_{:}_x123.{:}".format(model.rnnChaosName, mode, FIGTYPE)
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        axes[0].plot(tpts, prediction[:,0], color=color_dict['deepskyblue'], linewidth=2.0, label='prediction')
        axes[0].plot(tpts, target[:,0], color=color_dict['tomato'], linewidth=2.0, label='target')
        axes[0].set_ylabel(r"$x_1$")
        axes[0].legend(bbox_to_anchor=(0.25, 1.32), loc="upper left", ncol=2, frameon=False)

        axes[1].plot(tpts, prediction[:,1], color=color_dict['deepskyblue'], linewidth=2.0)
        axes[1].plot(tpts, target[:,1], color=color_dict['tomato'], linewidth=2.0)
        axes[1].set_ylabel(r"$x_2$")

        axes[2].plot(tpts, prediction[:,2], color=color_dict['deepskyblue'], linewidth=2.0)
        axes[2].plot(tpts, target[:,2], color=color_dict['tomato'], linewidth=2.0)
        axes[2].set_ylabel(r"$x_3$")

        if lambda_max is not None:
            axes[-1].set_xlabel(r"$\Lambda_1 t$")

        # plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()











