'''
Created on 11.10.2019

@author: Moritz
'''

import numpy as np
from matplotlib import pyplot as plt



def _set_standard(ax, title=None, x_label=None, y_label=None, xlim=None, ylim=None, x_tick_labels=None, y_tick_labels=None):
    if title is not None : ax.set_title(title)
    if x_label is not None : ax.set_xlabel(x_label)
    if y_label is not None : ax.set_ylabel(y_label)
    if xlim is not None : ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None : ax.set_ylim(ylim[0], ylim[1])
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticks(np.arange(len(y_tick_labels)))
        ax.set_yticklabels(y_tick_labels)



def bar_plot(ax, y_vals, x_tick_labels=None, y_err=None, title=None, x_label=None, y_label=None, xlim=None, ylim=None):
    ax.bar(np.arange(len(y_vals)), y_vals, yerr=y_err)  
    _set_standard(ax, title, x_label, y_label, xlim, ylim, x_tick_labels)
    
    
      
def multiple_bar_plot(ax, arr_vals, x_tick_labels=None, legend_labels=None, y_errs=None, title=None, x_label=None, y_label=None, xlim=None, ylim=None):
    if y_errs is None: y_errs = [None] * len(arr_vals)
    if legend_labels is None: legend_labels = [None] * len(arr_vals)
    n_bars = len(arr_vals)
    gap_size = 1/(n_bars+1)
    spacing_x = np.linspace(-0.5, 0.5, num=n_bars+1, endpoint=False)
    for i, vals in enumerate(arr_vals) : ax.bar(np.arange(len(vals))+spacing_x[i], vals, width=gap_size, yerr=y_errs[i], label=legend_labels[i])        
    _set_standard(ax, title, x_label, y_label, xlim, ylim, x_tick_labels)



def line_plot(ax, x_vals, y_vals, y_errs=None, label=None, title=None, x_label=None, y_label=None, xlim=None, ylim=None):
    ax.plot(x_vals, y_vals, label=label)
    if y_errs is not None: ax.fill_between(x_vals, y_vals-y_errs, y_vals+y_errs, alpha=0.5)
    _set_standard(ax, title, x_label, y_label, xlim, ylim)



def heatmap_plot(ax, data, x_labels, y_labels, value_display=False, title=None, x_label=None, y_label=None):
    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)
    
    if value_display:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, data[i, j], ha="center", va="center", color="w")
    
    _set_standard(ax, title, x_label, y_label, x_tick_labels=x_labels, y_tick_labels=y_labels)





        

if __name__ == '__main__':
    
    
    fig, axes = plt.subplots(3, 2, squeeze=False)
    
    
    bar_plot(axes[0][0], [1,2,3], ["a", "b", "c"], None, "Ein Diagramm", "x-label", "y-label")
    bar_plot(axes[1][0], [1,2,3], ["a", "b", "c"], y_err=[1,1,1], ylim=[0,4])
    
    multiple_bar_plot(axes[1][1], [[1,2,3],[3,2,1],[3,2,1],[3,2,1]])#, ["a", "b", "c"])
    
    
    line_plot(axes[0][1], [1,2,3,4,5,6,7,8], [2,5,7,3,5,3,7,8])
    
    
    heatmap_plot(axes[2][0], np.array([[1,2,3],[4,5,6],[7,8,9]]), ["x1", "x2", "x3"], ["y1", "y2", "y3"])
    
    plt.show()
    
    
    
    
    
    