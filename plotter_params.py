import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from cycler import cycler


def plot_setup(width='thesis'):
    if width == 'thesis':
        width_pt = 345
        rcParams['axes.labelsize'] = 18
        rcParams['xtick.labelsize'] = 16
        rcParams['ytick.labelsize'] = 16
        rcParams['legend.fontsize'] = 16
        # width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'pnas':
        width_pt = 246.09686
    elif width == 'physrev':
        width_pt = 253.125
    else:
        width_pt = width
    # Width of figure
    x = width_pt/72
    if width != 'thesis':
        rcParams['axes.labelsize'] = 16
        rcParams['xtick.labelsize'] = 14
        rcParams['ytick.labelsize'] = 14
        rcParams['legend.fontsize'] = 14
    rcParams['xtick.direction'] = 'inout'
    rcParams['ytick.direction'] = 'inout'
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.xmargin'] = 0.0
    rcParams['axes.ymargin'] = 0.05
    rcParams['legend.frameon'] = False
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['figure.figsize'] = 1.61803398875*x, x
#     rcParams['axes.prop_cycle'] =


if __name__ == "__main__":
    plot_setup()
    plt.plot([2*np.pi*(i/100) for i in range(100)], [np.sin(2*np.pi*(i/100))
                                                     for i in range(100)], label=r'$sin(\theta)$')
    plt.plot([2*np.pi*(i/100) for i in range(100)], [np.cos(2*np.pi*(i/100))
                                                     for i in range(100)], label=r'$cos(\theta)$')
    plt.legend()
    plt.ylabel(r'$sin(\theta)$')
    plt.xlabel(r'$\theta$')
    plt.show()
