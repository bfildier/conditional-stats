"""@package conditionalstats
Documentation for module conditionalstats.

Class Distribution defines objects containing the statistical distribution of a 
single variable, choosing the bin type (linear, logarithmic, inverse-logarithmic,
...).
"""
import numpy as np
from math import log10,ceil,floor,exp
import time
import sys

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
        - nlb: 'number of linear bins' used for linear statistics. Default is 50.
        - nbpd: number of bins per (log or invlog) decade. Default is 10.
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
        fill_last_decade=False,distribution=None,overwrite=False):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - bintype [linear, log, invlog]: bin structure,
        - nlb: number of bins used for linear statistics. Default is 50.
        - nlr: number of ranks in linQ bintype. Default is 100.
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
        self.overwrite = overwrite

        if distribution is not None: # then copy it in self
            for attr in distribution.__dict__.keys():
                setattr(self,attr,getattr(distribution,attr)) 

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< Distribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

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

    def computePercentilesAndBinsFromRanks(self,sample,crop=True,store=True):

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
        # print(centers)
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
        nbins = breaks.size - 1

        if store:
            self.percentiles = centers
            self.bins = breaks
            self.nbins = nbins # ()'bins' is actually bin edges)
        else:
            return centers, breaks, nbins

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

        if not self.overwrite:
            pass

        # Compute ranks, bins and percentiles
        self.ranksPercentilesAndBins(sample,vmin,vmax,minmode)
        # Compute probability density
        density, _ = np.histogram(sample,bins=self.bins,density=True)
        self.density = density
        # Number fraction of points below chosen vmin
        self.frac_below_vmin = np.sum(sample < self.vmin)/np.size(sample)
        # Number fraction of points above chosen vmax
        self.frac_above_vmax = np.sum(sample > self.vmax)/np.size(sample)

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
        - sample: here we assume data is horizontal, formats it in shape (Ncolumns,)
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

    def storeSamplePoints(self,sample,sizemax=50,verbose=False,method='shuffle_mask'):

        """Find indices of bins in the sample data, to go back and fetch later
        """

        if self.bin_locations_stored and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")

        # print(sample.shape)
        sample = self.formatDimensions(sample)
        # print(sample.shape)

        # Else initalize and find bin locations
        self.bin_locations = [[] for _ in range(self.nbins)]
        self.bin_sample_size = [0 for _ in range(self.nbins)]

        if method == 'random':

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
            
        elif method == 'shuffle_mask':

            if verbose: print('bin #: ',end='')
            # compute mask for each bin, randomize and store 'sizemax' first indices
            for i_bin in range(self.nbins):

                if verbose: print('%d..'%i_bin,end='')

                # compute mask
                mask = np.logical_and(sample.flatten() >= self.bins[i_bin],
                            sample.flatten() < self.bins[i_bin+1])
                # get all indices
                ind_mask = np.where(mask)[0]
                # shuffle
                np.random.seed(int(round(time.time() * 1000)) % 1000)
                np.random.shuffle(ind_mask)
                # select 'sizemax' first elements
                self.bin_locations[i_bin] = ind_mask[:sizemax]
                self.bin_sample_size[i_bin] = min(ind_mask.size,sizemax)


            if verbose: print()

        if verbose:
            print()

        # If reach this point, everything should have worked smoothly, so:
        self.bin_locations_stored = True

    def computeIndividualPercentiles(self,sample,ranks,out=False):
        """Computes percentiles of input sample and store in object attribute"""

        if isinstance(ranks,float) or isinstance(ranks,int):
            ranks = [ranks]
        
        result = []

        for r in ranks:
            # calculate percentile
            p = np.percentile(sample,r)
            result.append(p)
            # save
            setattr(self,"perc%2.0f"%r,p)

        if out:
            return result

    def computeInvCDF(self,sample,out=False):
        """Calculate inverse CDF on IL ranks: fraction of rain mass falling 
        above each percentile"""
        
        self.invCDF = np.ones(self.nbins)*np.nan
        sample_sum = np.nansum(sample)
        for iQ in range(self.nbins):
            rank = self.ranks[iQ]
            perc = self.percentiles[iQ]
            if not np.isnan(perc):
                self.invCDF[iQ] = np.nansum(sample[sample>perc])/sample_sum

        if out:
            return self.invCDF

    def bootstrapPercentiles(self,sample,nd_resample=10,n_bootstrap=50):
        """Perform bootstrapping to evaluate the interquartile range around each
        percentile, for the ranks stored.

        Arguments:
        - sample: np array in Nt,Ny,Nx format
        - nd_resample: number of time indices to randomly select for resampling
        - n_boostrap: number of times to calculate the distribution
        """

        sshape = sample.shape
        d_time = 0

        # calculate and store distribution n_bootstrap times
        perc_list = []
        for i_b in range(n_bootstrap):

            # select days randomly
            indices = list(range(sshape[d_time]))
            np.random.shuffle(indices)
            ind_times = indices[:nd_resample]
            resample = np.take(sample,ind_times,axis=d_time)

            # calculate percentiles on resample
            centers, bins, bins = self.computePercentilesAndBinsFromRanks(resample,
                                            crop=False,store=False)

            perc_list.append(centers)

        # combine distributions into statistics and save
        perc_array = np.vstack(perc_list)
        self.percentiles_sigma = np.std(perc_array,axis=0)
        self.percentiles_P5 = np.percentile(perc_array,5,axis=0)
        self.percentiles_Q1 = np.percentile(perc_array,25,axis=0)
        self.percentiles_Q2 = np.percentile(perc_array,50,axis=0)
        self.percentiles_Q3 = np.percentile(perc_array,75,axis=0)
        self.percentiles_P95 = np.percentile(perc_array,95,axis=0)
        
    def getCDF(self):
        """Compute the cumulative density function from the probability density,
        as: fraction pf points below vmin + cumulative sum of density*bin_width
        Output is the probability of x < x(bin i), same size as bins (bin edges)"""
        
        # array of bin widths
        bin_width = np.diff(self.bins)
        # CDF from density and bin width
        cdf_base = np.cumsum(bin_width*self.density)
        # readjust  to account for the fraction of points outside the range [vmin,vmax]
        fmin = self.frac_below_vmin
        fmax = self.frac_above_vmax
        cdf = fmin + np.append(0,cdf_base*(1-fmin-fmax))
        
        return cdf


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

    def __repr__(self):
        """Creates a printable version of the ConditionalDistribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< ConditionalDistribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

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
                    # nlev = sample_out.shape[0]
                    sample_out = np.swapaxes(sample_out,0,1)
                    sshape = sample_out.shape
                sample_out = np.reshape(sample_out,(sshape[0],np.prod(sshape[1:])))
            Nz,Npoints = sample_out.shape
            # if self.isTime:
            #     Npoints = Npoints/nlev
            #     print(Npoints)
        else:
            if len(sshape) > 1: # reshape
                sample_out = np.reshape(sample,np.prod(sshape))
            Npoints, = sample_out.shape
            Nz = None
        
        # Test if sample size is correct to access sample points
        if Npoints != self.on.size:
            # print(Npoints,self.on.size)
            raise WrongArgument("ABORT: sample size is different than that of the reference variable, so the masks might differ")

        return Nz, sample_out

    def computeConditionalMeanAndVariance(self,sample,verbose=False):
        """Computes mean and variance of input data at each percentile of 
        reference variable (in bins self.on.bins).

        Arguments:
        - sample: if is3D, should be in format (Nz,Ncolumns), otherwise (Ncolumns,)
        If not, it is formatted using method formatDimensions.
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

class DistributionOverTime(Distribution):
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

    def __repr__(self):
        """Creates a printable version of the DistributionOverTime object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< DistributionOverTime object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

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

    def computeDistributions(self,sample,**kwargs):
        """Fills up the distribution of timeVar 

        Arguments:
        - sample: np.array of dimensions (nt,...)
        - *args: see input parameters in method Distribution.computeDistribution
        """

        # Test dimensions
        self.testInput(sample)

        # Compute all distributions over time
        for it_slice,it_store in self.iterRefTimeIndices():

            self.distributions[it_store].computeDistribution(sample[it_slice],**kwargs)

    def storeSamplePoints(self,sample,sizemax=50,method='shuffle_mask',verbose=False):
        """Find indices of bins in the sample data, to go back and fetch
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            print("%d_%d"%(it_slice.start,it_slice.stop),end=' ; ')

            self.distributions[it_store].storeSamplePoints(sample=sample[it_slice],sizemax=sizemax,method=method,verbose=verbose)

        print()

    def computeIndividualPercentiles(self,sample,ranks):
        """Computes percentiles of input sample and store timeseries in object
        attribute. CAREFUL here only do calculation at each time, without using
        iterRefTimeIndices method.
        
        Arguments:
        - sample as above
        - ranks: float, list or np.array"""

        # Test dimensions
        self.testInput(sample)

        if isinstance(ranks,float):
            ranks = [ranks]
        
        for r in ranks:

            vals = np.nan*np.zeros((self.nt,))

            # Compute all distributions over time
            for it_slice,it_store in self.iterRefTimeIndices():

                vals[it_store] = self.distributions[it_store].computeIndividualPercentiles(sample[it_slice],r,out=True)[0]

            # save
            setattr(self,"perc%2.0f"%r,vals)

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

    def __repr__(self):
        """Creates a printable version of the ConditionalDistributionOverTime object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< ConditionalDistributionOverTime object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

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
            
    def storeSamplePoints(self,sample,sizemax=50,method='shuffle_mask',verbose=False):
        """Find indices of bins in the sample data, to go back and fetch

        Arguments:
        - sample: reference sample! np.array of dimensions (nt,...)
        - sizemax: maximum number of indices stored for each bin
        """

        # Test dimensions
        self.testInput(sample)

        # Compute bin locations if not known already
        for it_slice,it_store in self.iterRefTimeIndices():

            print("%d_%d"%(it_slice.start,it_slice.stop),end=' ; ')

            self.cond_distributions[it_store].on.storeSamplePoints(sample=sample[it_slice],\
                sizemax=sizemax,method=method,verbose=verbose)

        print()

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
        