"""Module plot1DInvLog

Functions to plot curves on inverse-logarithmic x-axis (extremes).
"""


#---- Modules ----#

import matplotlib.pyplot as plt
import numpy as np
from math import log10,ceil
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm


#---- Functions ----#

def transformXaxisIL(ax,x,offset=0):
    
    # reverse x-axis
    ax.invert_xaxis()
    # rename ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    n = ceil(log10(x.max()))
    N = len(labels)
    for i in range(1,N):
        labels[-n+i-4+offset] = str(100*(1-10**(-n+i-1)))
        if -n+i-1 == 0:
            break
    ax.set_xticklabels(labels)

def transformYaxisIL(ax,y,offset=0):
    
    # reverse y-axis
    ax.invert_yaxis()
    # rename ticks
    labels = [item.get_text() for item in ax.get_yticklabels()]
    n = ceil(log10(y.max()))
    N = len(labels)
    for i in range(1,N):
        labels[-n+i-4+offset] = str(100*(1-10**(-n+i-1)))
        if -n+i-1 == 0:
            break
    ax.set_yticklabels(labels)

def subplotRanksILog(ax,ranks,y,col=None,ltype=None,linewidth=None,alpha=None,
    labels=None,rankmin=0,rankmax=100,flipx=True,transformX=False):
    
    ax.set_xscale('log')

    # mask2plot = np.logical_and(ranks>rankmin,ranks<rankmax)
    # define x-axis
    # x = np.flipud(1./(1-ranks[mask2plot]/100.))
    # print("%1.2f"%(np.sum(mask2plot)/len(mask2plot)),ranks[mask2plot])
    # plot
    if isinstance(y,list):
        for i in range(len(y)):
            lab = labels[i] if labels is not None else None
            lt = ltype[i] if ltype is not None else '-'
            a = alpha[i] if alpha is not None else 1
            c = col[i] if col is not None else 1
            lw = linewidth[i] if linewidth is not None else 1.5
            # define x-axis
            mask2plot = np.logical_and(ranks[i]>rankmin,ranks[i]<rankmax)
            x = 1./(1-ranks[i][mask2plot]/100.)
            if flipx:
                x = np.flipud(x)
            # plot
            ax.plot(x,y[i][mask2plot],c=c,alpha=a,linestyle=lt,linewidth=lw,label=lab)
            # ax.plot(x,np.flipud(y[i][mask2plot]),c=c,alpha=a,linestyle=lt,linewidth=lw,label=lab)
    else:
        # define x-axis
        mask2plot = np.logical_and(ranks>rankmin,ranks<rankmax)
        x = 1./(1-ranks[mask2plot]/100.)
        if flipx:
            x = np.flipud(x)
        # plot
        ax.plot(x,y[mask2plot],c=col,alpha=alpha,linestyle=ltype,linewidth=linewidth,label=labels)
        # ax.plot(x,np.flipud(y[mask2plot]),c=col,alpha=alpha,linestyle=ltype,linewidth=linewidth,label=labels)

    # transform x-axis
    if transformX:
        transformXaxisIL(ax,x)
    
def subplotYShadingRanksILog(ax,ranks,y_BCs,col,alpha=0.2,transformX=False):
    
    ax.set_xscale('log')
    
    # define x-axis
    x = np.flipud(1./(1-ranks/100.))
    # plot
    y1 = y_BCs[0]
    y2 = y_BCs[1]
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor=col,alpha=alpha, interpolate=True)
    
    # transform x-axis
    if transformX:
        transformXaxisIL(ax,x)

def subplotXShadingRanksILog(ax,ranks,iQ_lims,alpha=0.2,col='0.75',transformX=False):

    ax.set_xscale('log')
    
    # define x-axis
    x = np.flipud(1./(1-ranks/100.))
    if iQ_lims[0] >= x.size:
        return
    x0 = x[iQ_lims[0]]
    if iQ_lims[1] >= x.size:
        x1 = x[-1]
    else:
        x1 = x[iQ_lims[1]]
    # plot
    ax.axvspan(x0,x1,color = '0.75',alpha=alpha)
    
    # transform x-axis
    if transformX:
        transformXaxisIL(ax,x)

def addXHatchRanksILog(ax,ranks,iQ_lims,color='gray',hatch='//',
    alpha=1,transformX=False,fill=False):

    ax.set_xscale('log')

    x = np.flipud(1./(1-ranks/100.))
    ax.add_patch(Polygon([[x[iQ_lims[0]], ax.get_ylim()[0]],\
                          [x[iQ_lims[1]], ax.get_ylim()[0]],\
                          [x[iQ_lims[1]], ax.get_ylim()[1]],\
                          [x[iQ_lims[0]], ax.get_ylim()[1]]],\
                          closed=True, fill=fill, hatch=hatch,linewidth=0,
                          color=color,alpha=alpha))
    # transform x-axis
    if transformX:
        transformXaxisIL(ax,x)

def addZeroLine(ax,x,col='gray',alpha=1,transformX=False):

    ax_line = ax.twinx()
    subplotRanksILog(ax_line,x,
                     np.zeros(x.size),
                     col=col,alpha=alpha,ltype='-',linewidth=0.8,transformX=transformX)
    ax_line.yaxis.set_ticks_position('none')
    ax_line.yaxis.set_ticklabels('')
    ax_line.set_ylim(ax.get_ylim())

def addYLine(ax,x,y0=0,c='gray',lt='-',lw=0.8,transformX=False):

    ax_line = ax.twinx()
    subplotRanksILog(ax_line,x,
                     y0*np.ones(x.size),
                     col=c,ltype=lt,linewidth=lw,transformX=transformX)
    ax_line.yaxis.set_ticks_position('none')
    ax_line.yaxis.set_ticklabels('')
    ax_line.set_ylim(ax.get_ylim())
    
def add1to1Line(ax):

    x = ax.get_xlim()
    y = ax.get_ylim()

    xmin = min(x[0],y[0])
    xmax = max(x[1],y[1])

    ax.plot([xmin,xmax],[xmin,xmax],'k',lw=1)

def highlightPointRanksILog(ax,pt):

    x_pt = 1./(1-pt[0]/100)
    y_pt = pt[1]
    # y axis, linear
    ylims = ax.get_ylim()
    ymax = (y_pt-ylims[0])/(ylims[1]-ylims[0])
    ax.axvline(x=x_pt,ymax=ymax,linewidth=0.5,c='gray')
    # x axis, use log 
    xlims = ax.get_xlim()    
    xmax = (log(x_pt)-log(xlims[0]))/(log(xlims[1])-log(xlims[0]))
    ax.axhline(y=y_pt,xmax=xmax,linewidth=0.5,c='gray')
    # Draw point
    ax.scatter(x_pt,y_pt,marker='o',c='gray')


