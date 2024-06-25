import numpy as np
import socket
import os
import subprocess
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from scipy.stats.stats import pearsonr
from .global_utils import *
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


FIGTYPE = "pdf"


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


def PlotPredictions(model, mode, target, prediction, dt, lambda_max=None, plotEmbed=False, plotContour=False):
    if lambda_max is None:
        tpts = dt * np.arange(np.shape(prediction)[0])
    else:
        tpts = dt * np.arange(np.shape(prediction)[0]) * lambda_max

    if plotContour:
        PlotTestContours(model, mode, target, prediction, tpts, dt)
    else:
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
            fig_path = model.getFigureDir() + "/{:}_{:}_x12.{:}".format(model.rnnChaosName, mode, FIGTYPE)
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            axes[0].plot(tpts, prediction[:,0], color=color_dict['deepskyblue'], linewidth=1.5, label='prediction')
            axes[0].plot(tpts, target[:,0], color=color_dict['tomato'], linewidth=1.5, label='target')
            axes[0].set_ylabel(r"$x_1$")
            axes[0].legend(bbox_to_anchor=(0.25, 1.32), loc="upper left", ncol=2, frameon=False)

            axes[1].plot(tpts, prediction[:,1], color=color_dict['deepskyblue'], linewidth=1.5)
            axes[1].plot(tpts, target[:,1], color=color_dict['tomato'], linewidth=1.5)
            axes[1].set_ylabel(r"$x_2$")

            if lambda_max is not None:
                axes[-1].set_xlabel(r"$\Lambda_1 t$")

            # plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
        

def PlotTestContours(model, mode, target, prediction, tpts, dt):

    error = np.abs(target-prediction)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 10), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    contours_vec = []

    t0, x0 = np.meshgrid(tpts, np.arange(target.shape[1]))
    c0 = axes[0].pcolor(x0, t0, target.transpose(), shading='auto', cmap=cc.cm['diverging_bwr_55_98_c37'])
    axes[0].set_title('Target')
    axes[0].set_ylabel(r"$\Lambda_1 t$")
    axes[0].set_xlabel(r"$x$")
    contours_vec.append(c0)

    t1, x1 = np.meshgrid(tpts, np.arange(prediction.shape[1]))
    c1 = axes[1].pcolor(x1, t1, prediction.transpose(), shading='auto', cmap=cc.cm['diverging_bwr_55_98_c37'])
    axes[1].set_title('Prediction')
    axes[1].set_xlabel(r"$x$")
    contours_vec.append(c1)

    t2, x2 = np.meshgrid(tpts, np.arange(error.shape[1]))
    c2 = axes[2].pcolor(x2, t2, error.transpose(), shading='auto', cmap=cc.cm['diverging_bwr_55_98_c37'])
    axes[2].set_title('Error')
    axes[2].set_xlabel(r"$x$")
    contours_vec.append(c2)

    fig.colorbar(c2, ax=axes[2])

    fig_path = model.getFigureDir() + "/{:}_{:}_contour.{:}".format(model.rnnChaosName, mode, FIGTYPE)
    plt.savefig(fig_path)
    plt.close()


        


