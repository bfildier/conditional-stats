"""@package conditionalstats
Documentation for module conditionalstats.

Class Distribution defines objects containing the statistical distribution of a 
single variable, choosing the bin type (linear, logarithmic, inverse-logarithmic,
...).
"""
import numpy as np
from math import log10,ceil,floor,exp

class WrongArgument(Exception):
    pass

class EmptyDataset(Exception):
    pass

class EmptyDistribution:

    """Documentation for class EmptyDistribution

    Base object
    """

    def __init__(self,bintype='linear',nbpd=10,nppb=4,nlb=50,nlr=100,fill_last_decade=False):

        """Constructor for class EmptyDistribution.
        Arguments:
        - bintype [linear, log, invlogQ, linQ]: bin structure.
        - nlb: number of bins used for linear statistics. Default is 50.
        - nbpd: number of bins per log or invlog decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nlr: number of ranks in linQ bintype. Default is 100.
        - fill_last_decade: bool to fill up largest percentiles for invlog bintype
        """

        self.bintype = bintype
        self.nlb = nlb
        self.nbpd = nbpd
        self.nppb = nppb
        self.nlr = nlr
        self.fill_last_decade = fill_last_decade

        if self.bintype == 'linear':

            self.nbpd = None
            self.nppb = None
            self.nlr = None
            self.fill_last_decade = None

        elif self.bintype in ['log','invlogQ']:

            self.nlb = None
            self.nlr = None

        elif self.bintype == 'linQ':

            self.nlb = None
            self.nbpd = None
            self.nppb = None
            self.fill_last_decade = None

        else:

            raise WrongArgument("ERROR: unknown bintype")
    
    def __str__(self):
        """Override string function to print attributes
        """
        # method_names = []
        # str_out = '-- Attributes --'
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
        #         else:
        #             method_names.append(a)
        # print('-- Methods --')
        # for m in method_names:
        #     print(m)
        return str_out


class Distribution(EmptyDistribution):
    """Documentation for class Distribution
    """

    def __init__(self,name='',bintype='linear',nbpd=10,nppb=4,nlb=50,nlr=100,\
        fill_last_decade=False):

        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - bintype [linear, log, invlog]: bin structure,
        - nlb: number of bins used for linear statistics. Default is 50.
        - nbpd: number of bins per log or invlog decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        """

        EmptyDistribution.__init__(self,bintype,nbpd,nppb,nlb,nlr,fill_last_decade)
        self.name = name
        self.size = 0
        self.vmin = None
        self.vmax = None
        self.ranks = None
        self.percentiles = None
        self.bins = None
        self.density = None

    def __str__(self):
        return super().__str__()

    def setSampleSize(self,sample):

        if sample.size == 0:
            raise EmptyDataset("")
        else:
            self.size = sample.size

    def setVminVmax(self,sample=None,vmin=None,vmax=None,minmode='positive',\
        overwrite=False):

        """Compute and set minimum and maximum values
        Arguments:
        - sample: 1D numpy array of data values."""

        # Find minimum value
        if vmin is None:	
            if minmode is None:
                vmin = np.nanmin(sample)
            elif minmode == 'positive':
                vmin = np.nanmin(sample[sample > 0])
        # Find maximum value
        if vmax is None:
            vmax = np.nanmax(sample)
            
        if self.vmin is None or overwrite:
            self.vmin = vmin
        if self.vmax is None or overwrite:
            self.vmax = vmax

    def getInvLogRanks(self):

        """Percentile ranks regularly spaced on an inverse-logarithmic axis (zoom on 
        largest percentiles of the distribution).
        Arguments:
            - fill_last_decade: True (default is False) if want to plot
            up to 99.99 or 99.999, not some weird number in the middle of a decade.
        Sets:
            - ranks: 1D numpy.array of floats"""

        # k indexes bins
        n_decades = log10(self.size/self.nppb) 		# Maximum number of decades
        dk = 1/self.nbpd
        if self.fill_last_decade:
            k_max = floor(n_decades)				 	# Maximum bin index
        else:
            k_max = int(n_decades*self.nbpd)*dk # Maximum bin index
        scale_invlog = np.arange(0,k_max+dk,dk)
        ranks_invlog = np.subtract(np.ones(scale_invlog.size),
            np.power(10,-scale_invlog))*100

        self.ranks = ranks_invlog

    def getLinRanks(self):

        """Percentile ranks regularly spaced on a linear axis of percentile ranks"""

        self.ranks = np.linspace(0,100,self.nlr+1)

    def computePercentilesAndBinsFromRanks(self,sample,crop=True):

        """Compute percentiles of the distribution and histogram bins from 
        percentile ranks. 
        Arguments:
            - sample: 1D numpy array of values
            - ranks: 1D array of floats between 0 and 1
        Sets:
            - ranks, cropped by one at beginning and end
            - percentiles (or bin centers)
            - bins (edges)
        """

        sample_no_nan = sample[np.logical_not(np.isnan(sample))]
        if sample_no_nan.size == 0:
            centers = np.array([np.nan]*self.ranks.size)
        else:
            centers = np.percentile(sample_no_nan,self.ranks)

        breaks = np.convolve(centers,[0.5,0.5],mode='valid')
        if crop:
            centers = centers[1:-1]
            self.ranks = self.ranks[1:-1]
        else:
            temp = breaks.copy()
            breaks = np.array([np.nan]*(temp.size+2))
            breaks[0] = self.vmin
            breaks[1:-1] = temp
            breaks[-1] = self.vmax

        self.percentiles = centers
        self.bins = breaks

    def definePercentilesOnInvLogQ(self,sample):

        """Defines percentiles and histogram bins on inverse-logarithmic ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """
        
        self.size = sample.size
        # First compute invlog ranks including its edge values
        self.getInvLogRanks()
        # Then compute final stats
        self.computePercentilesAndBinsFromRanks(sample)

    def definePercentilesOnLinQ(self,sample,vmin=None,vmax=None):

        """Define percentiles and histogram bins on linear ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """

        self.setVminVmax(sample=sample,vmin=vmin,vmax=vmax)
        # Compute linear ranks
        self.getLinRanks()
        # Then compute final stats
        self.computePercentilesAndBinsFromRanks(sample,crop=False)

    def defineLogBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define logarithmic bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - n_bins_per_decade: number of ranks/bins per logarithmic decade
            - vmin and vmax: extremum values
        Computes:
            - centers (corresponding percentiles, or bin centers)
            - breaks (histogram bin edges)"""

        self.setVminVmax(sample,vmin,vmax,minmode)
        kmin = floor(log10(self.vmin))
        kmax = ceil(log10(self.vmax))
        # dk = 1/self.nbpd
        # exps = np.arange(kmin,kmax+dk,dk)
        # self.bins = np.power(10.,exps)
        # self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
    
        self.bins = np.logspace(kmin,kmax,(kmax-kmin)*self.nbpd)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        # exp_bins = np.arange(kmin,kmax+dk,dk)
        # exp_centers = np.convolve(exp_bins,[0.5,0.5],mode='valid')
        # self.bins = np.power(10.,exp_bins)
        # self.percentiles = np.power(10.,exp_centers)

    def defineLinearBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define linear bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - vmin and vmax: extremum values
        Computes:
            - percentiles (or bin centers)
            - bins (edges)
        """

        self.setVminVmax(sample,vmin,vmax,minmode)
        self.bins = np.linspace(self.vmin,self.vmax,self.nlb)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')

    def computePercentileRanksFromBins(self,sample):

        """Computes percentile ranks corresponding to percentile values.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks: 1D numpy.ndarray"""
        
        self.ranks = 100*np.array(list(map(lambda x:(sample < x).sum()/self.size, \
            self.percentiles)))

    def ranksPercentilesAndBins(self,sample,vmin=None,vmax=None,minmode='positive',\
        crop=True):

        """Preliminary step to compute probability densities. Define 
        ranks, percentiles, bins from the sample values and binning structure.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles and bins"""

        self.setSampleSize(sample)
        self.setVminVmax(sample,vmin,vmax,minmode)

        if self.bintype == 'linear':

            self.defineLinearBins(sample,vmin,vmax,minmode)
            self.computePercentileRanksFromBins(sample)

        elif self.bintype == 'log':

            self.defineLogBins(sample,vmin,vmax,minmode)
            self.computePercentileRanksFromBins(sample)

        elif self.bintype == 'invlogQ':

            self.getInvLogRanks()
            self.computePercentilesAndBinsFromRanks(sample,crop=crop)

        elif self.bintype == 'linQ':

            self.definePercentilesOnLinQ(sample)

        else:

            raise WrongArgument("ERROR: unknown bintype")

    def computeDistribution(self,sample,vmin=None,vmax=None,minmode=None):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample: 1D numpy array of values
        Returns:
            - ranks, percentiles, bins and probability densities"""

        # Compute ranks, bins and percentiles
        self.ranksPercentilesAndBins(sample,vmin,vmax,minmode)
        # Compute probability density
        density, _ = np.histogram(sample,bins=self.bins,density=True)
        self.density = density


class ConditionalDistribution():
"""Class ConditionalDistribution.

Stores conditional mean and variance in bins of a reference distribution.
"""

    def __init__(self,name='',refDistribution=None):
        """Contructor
        """

        self.name = name
        self.referenceDistribution = refDistribution
        
