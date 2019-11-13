
from scipy.interpolate import griddata,interp2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ## Plot vertical data (transect, or vertical profiles over time)
# def subplotVerticalData(ax,x,y,Z,cmap=plt.cm.seismic,vmin=None,vmax=None,cbar=True):
    
#     """Arguments:
#         - x and y are coordinate values
#         - Z are the data values flattened"""
    
#     xs0,ys0 = np.meshgrid(x,y)
#     xmin = min(x[0],x[-1])
#     xmax = max(x[0],x[-1])
#     ymin = min(y[0],y[-1])
#     ymax = max(y[0],y[-1])
#     X0 = np.vstack([xs0.flatten(),ys0.flatten()]).T
  
#     extent = (xmax,xmin,ymin,ymax)

#     # New coordinates
#     xs,ys = np.meshgrid(np.linspace(xmin,xmax,num=len(x)),
#                         np.linspace(ymin,ymax,num=len(y)))
#     X = np.vstack([xs.flatten(),ys.flatten()]).T

#     # New values
#     resampled = griddata(X0,Z,X, method='cubic')
#     resampled_2D = np.reshape(resampled,xs.shape)

#     im = ax.imshow(np.flipud(resampled_2D),extent=extent,interpolation='bilinear',
#                    cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto',origin='upper')
#     if cbar:
#         plt.colorbar(im,ax=ax)
    
#     if y[0] > y[-1]:
#         plt.gca().invert_yaxis()


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def subplotSmooth2D(ax,x,y,Z,fplot='contourf',xmin=None,xmax=None,nx=50,nlev=50,vmin=None,vmax=None,**kwargs):
    """
    Plot 2D contours (exact method is defined by fplot) with user-defined Z-range and x range.
    """
    
    # set levels
    levels = nlev
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin,vmax,nlev+1)
    
    # set new x plotting range and interpolate Z onto it
    x_new = x.copy()
    Z_new = Z.copy()
    if xmin is not None and xmax is not None:
        x_new = np.linspace(xmin,xmax,nx+1)
        f_interp = interp2d(x,y,Z,kind='cubic')
        Z_new = f_interp(x_new,y)
    X,Y = np.meshgrid(x_new,y)
    
    # remove values outside xrange
    X_out = np.logical_or(X < np.min(x),X > np.max(x))
    Z_new[X_out] = np.nan
    
    # plot
    return getattr(ax,fplot)(X,Y,Z_new,levels=levels,**kwargs)