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
        self.nbins = 0
        self.vmin = None
        self.vmax = None
        self.ranks = None
        self.percentiles = None
        self.bins = None
        self.density = None
        self.bin_locations_stored = False

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
        self.nbins = self.ranks.size

    def getLinRanks(self):

        """Percentile ranks regularly spaced on a linear axis of percentile ranks"""

        self.ranks = np.linspace(0,100,self.nlr+1)
        self.nbins = self.ranks.size

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
            centers = np.array([np.nan]*self.nbins)
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
        self.nbins = self.bins.size - 1 # ()'bins' is actually bin edges)

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
        self.bins = np.logspace(kmin,kmax,(kmax-kmin)*self.nbpd)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        self.nbins = self.percentiles.size

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
        self.bins = np.linspace(self.vmin,self.vmax,self.nlb+1)
        self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        self.nbins = self.percentiles.size

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

    def indexOfRank(self,rank):
    
        """Returns the index of the closest rank in numpy.array ranks"""

        dist_to_rank = np.absolute(np.subtract(self.ranks,rank*np.ones(self.ranks.shape)))
        mindist = dist_to_rank.min()
        return np.argmax(dist_to_rank == mindist)

    def rankID(self,rank):

        """Convert rank (float) to rank id (string)
        """

        return "%2.4f"%rank

    def binIndex(self,percentile=None,rank=None):

        """Returns the index of bin corresponding to percentile or rank 
        of interest
        """

        if percentile is not None:
            # Find first bin edge to be above the percentile of interest
            i_perc = np.argmax(self.bins > percentile)
            if i_perc == 0: # Then percentile is outside of stored bins
                return None
            return i_perc-1 # Offset by 1

        if rank is not None:
            return self.indexOfRank(rank)
        # raise WrongArgument("no percentile or rank is provided in binIndex")
        return None

    def formatDimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.

        Arguments:
        - sample: if is3D, format it in shape (Nz,Ncolumns), otherwise (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """

        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if len(sshape) > 1: # reshape
            sample_out = np.reshape(sample,np.prod(sshape))
        Npoints, = sample_out.shape
        
        # Test if sample size is correct to access sample points
        if Npoints != self.size:
            raise WrongArgument("Error: used different sample size")

        return sample_out

    def storeSamplePoints(self,sample,sizemax=30,verbose=False):

        """Find indices of bins in the sample data, to go back and fetch later
        """

        if self.bin_locations_stored:
            pass

        if verbose:
            print("Finding bin locations...")

        # print(sample.shape)
        sample = self.formatDimensions(sample)
        # print(sample.shape)

        # Else initalize and find bin locations
        self.bin_locations = [[] for _ in range(self.nbins)]
        self.bin_sample_size = [0 for _ in range(self.nbins)]

        # Look at all points, in random order
        indices = list(range(self.size))
        np.random.shuffle(indices)

        bins_full = []
        for i_ind in range(len(indices)):

            i = indices[i_ind]

            # Find corresponding bin
            i_bin = self.binIndex(percentile=sample[i])
            
            # Store only if bin was found
            if i_bin is not None:
                
                # Keep count
                self.bin_sample_size[i_bin] += 1
                # Store only if there is still room in stored locations list
                if len(self.bin_locations[i_bin]) < sizemax:
                    self.bin_locations[i_bin].append(i)
                elif i_bin not in bins_full:
                    bins_full.append(i_bin)
                    bins_full.sort()
                    if verbose:
                        print("%d bins are full (%d iterations)"%(len(bins_full),i_ind))
        
        if verbose:
            print()

        # If reach this point, everything should have worked smoothly, so:
        self.bin_locations_stored = True

class ConditionalDistribution():
    """Documentation for class ConditionalDistribution.

    Stores conditional mean and variance in bins of a reference distribution.
    """

    def __init__(self,name='',is3D=False,isTime=False,on=None):
        """Contructor
        
        Arguments:
        - name
        - is3D: boolean, if variable is defined in the vertical dimension
        - on: Object Distribution containing stats of reference variable
        """

        self.name = name
        self.is3D = is3D
        self.isTime = isTime
        self.on = on
        self.mean = None
        self.cond_mean = None
        self.cond_var = None
        
    def computeMean(self,sample):
        """Computes the (full) mean of the input data
        """

        self.mean = np.nanmean(sample)

    def formatDimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.

        Arguments:
        - sample: if is3D, format it in shape (Nz,Ncolumns), otherwise (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """

        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if self.is3D:
            # collapse dimensions other than z
            if len(sshape) > 2: # reshape
                # if time dimension (in 2nd dimension), reorder to have z in first dim
                if self.isTime:
                    sample_out = np.swapaxes(sample_out,0,1)
                    sshape = sample_out.shape
                sample_out = np.reshape(sample_out,(sshape[0],np.prod(sshape[1:])))
            Nz,Npoints = sample_out.shape
        else:
            if len(sshape) > 1: # reshape
                sample_out = np.reshape(sample,np.prod(sshape))
            Npoints, = sample_out.shape
            Nz = None
        
        # Test if sample size is correct to access sample points
        if Npoints != self.on.size:
            raise WrongArgument("ABORT: sample size is different than that of the reference variable, so the masks might differ")

        return Nz, sample_out

    def computeConditionalMeanAndVariance(self,sample,verbose=False):
        """Computes mean and variance of input data at each percentile of 
        reference variable (in bins self.on.bins).

        Arguments:
        - sample: if is3D, must be in format (Nz,Ncolumns), otherwise (Ncolumns,)
        If not, format it using method formatDimensions.
        """

        # Abort if sample points for each percentile has not been computed yet
        if self.on is None or not self.on.bin_locations_stored:
            raise EmptyDataset("Abort: must calculate bin locations of reference distribution first")

        # format dataset and test validity of input dataset
        Nz, sample = self.formatDimensions(sample)

        # Initialize storing arrays
        if self.is3D:
            self.cond_mean = np.nan*np.zeros((Nz,self.on.nbins))
            self.cond_var = np.nan*np.zeros((Nz,self.on.nbins))
        else:
            self.cond_mean = np.nan*np.zeros((self.on.nbins,))
            self.cond_var = np.nan*np.zeros((self.on.nbins,))

        # Access sample points to calculate conditional stats
        # automate
        def apply2vector(fun,vector):
            out = np.nan*np.zeros(self.on.nbins)
            for i_b in range(self.on.nbins): # loop over bins
                subsample = np.take(vector,self.on.bin_locations[i_b])
                if subsample.size == 0:
                    if verbose:
                        print('passing bin %d, subsample of size %d'%(i_b,subsample.size))
                    # pass
                else:
                    if verbose:
                        print("bin %d, result:%2.2f"%(i_b,fun(subsample)))
                    out[i_b] = fun(subsample)
            return out
        # compute
        if self.is3D:
            for i_z in range(Nz): # loop over heights
                self.cond_mean[i_z] = apply2vector(np.nanmean,np.squeeze(sample[i_z]))
                self.cond_var[i_z] = apply2vector(np.nanvar,np.squeeze(sample[i_z]))
        else:
            self.cond_mean = apply2vector(np.nanmean,sample)
            self.cond_var = apply2vector(np.nanvar,sample)

        self.cond_std = np.sqrt(self.cond_var)

class DistributionOverTime():
    """Time evolution of an object of class Distribution.
    """

    def __init__(self,name='',time_ref=[],width=0,**kwargs):
        """Constructor of class DistributionOverTime

        Arguments:
        - *args: see input parameters of constructor Distribution.__init__
        - time_ref: np.array of time values for reference dataset (in days, usually)
        - time: np.array of time values for calculated statistics
        - width: width of time window used to calculate statistics (same unit as time)
        """
        # for key, value in kwargs.items(): 
        #     print ("%s == %s" %(key, value))
        self.name = name
        self.time_ref = time_ref    
        self.nt = len(self.time_ref)
        self.width = width
        # time step of reference data
        self.dt = 0
        if self.nt > 1:
            self.dt = np.diff(self.time_ref)[0]
        # remove dn first and last points for statistics
        self.dn = 0
        if self.dt != 0:
            self.dn = int(self.width/2./self.dt)
        # destination time values
        self.time = self.time_ref[self.dn:len(self.time_ref)-self.dn]
        # initialize empty distributions
        self.distributions = [Distribution(name,**kwargs) for i in range(self.nt-2*self.dn)]

    def iterTime(self):

        return range(self.nt-2*self.dn)

    def iterRefTimeIndices(self):

        ref_inds = range(self.dn,self.nt-self.dn)
        it_slices = [slice(i_t-self.dn,i_t+self.dn+1) for i_t in ref_inds]
        it_stores = [i_t-self.dn for i_t in ref_inds]

        return zip(it_slices,it_stores)

    def testInput(self,sample):
        """Test that time dimension is matching first dimension of imput sample
        """
        sshape = sample.shape
        if self.nt == 1:
            return
        if sshape[0] != self.nt:
            raise WrongArgument('ABORT: input sample does not have the correct'+\
            ' time dimension')

    def computeDistributions(self,sample,*args):
        """Fills up the distribution of timeVar 

        Arguments:
        - sample: np.array of dimensions (nt,...)
        - *args: see input parameters in method Distribution.computeDistribution
        """

        # Test dimensions
        self.testInput(sample)

        # Compute all distributions over time
        for it_slice,it_store in self.iterRefTimeIndices():

            self.distributions[it_store].computeDistribution(sample[it_slice],*args)

    def storeSamplePoints(self,sample,sizemax=50,verbose=False):
        """Find indices of bins in the sample data, to go back and fetch
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            self.distributions[it_store].storeSamplePoints(sample=sample[it_slice],sizemax=sizemax,verbose=verbose)

class ConditionalDistributionOverTime():
    """Time evolution of an object of class ConditionalDistribution.
    """

    def __init__(self,name='',time_ref=[],width=0,is3D=False,isTime=True,on=None):
        """Constructor of class ConditionalDistributionOverTime

        Arguments:
        - *args: see input parameters of constructor ConditionalDistribution.__init__
        """

        self.name = name
        self.time_ref = time_ref
        self.nt = len(self.time_ref)
        self.width = width
        # time step of reference data
        self.dt = 0
        if self.nt > 1:
            self.dt = np.diff(self.time_ref)[0]
        # remove dn first and last points for statistics
        self.dn = 0
        if self.dt != 0:
            self.dn = int(self.width/2./self.dt)
        # destination time values
        self.time = self.time_ref[self.dn:len(self.time_ref)-self.dn]

        self.cond_distributions = []
        # Initializes all reference distributions
        for on_i in on.distributions:
            self.cond_distributions.append(ConditionalDistribution(name,is3D=is3D,isTime=isTime,on=on_i))

    def iterTime(self):

        return range(self.nt-2*self.dn)
        
    def iterRefTimeIndices(self):

        ref_inds = range(self.dn,self.nt-self.dn)
        it_slices = [slice(i_t-self.dn,i_t+self.dn+1) for i_t in ref_inds]
        it_stores = [i_t-self.dn for i_t in ref_inds]

        return zip(it_slices,it_stores)

    def testInput(self,sample):
        """Test that time dimension is matching first dimension of imput sample
        """
        sshape = sample.shape
        if self.nt == 1:
            return
        if sshape[0] != self.nt:
            raise WrongArgument('ABORT: input sample does not have the correct'+\
            ' time dimension')
            
    def storeSamplePoints(self,sample,sizemax=50,verbose=False):
        """Find indices of bins in the sample data, to go back and fetch

        Arguments:
        - sample: reference sample! np.array of dimensions (nt,...)
        - sizemax: maximum number of indices stored for each bin
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            self.cond_distributions[it_store].on.storeSamplePoints(sample=sample[it_slice],sizemax=sizemax,verbose=verbose)

    def computeConditionalStatsOverTime(self,sample,**kwargs):
        """Fills up the distribution of timeVar 

        Arguments:
        - sample: np.array of dimensions (nt,...)
        - **kwargs: see input parameters in method ConditionalDistribution.computeConditionalMeanAndVariance
        """

        # Test dimensions
        self.testInput(sample)

        # Compute all distributions over time
        for it_slice,it_store in self.iterRefTimeIndices():

            self.cond_distributions[it_store].computeConditionalMeanAndVariance(sample[it_slice],**kwargs)
        