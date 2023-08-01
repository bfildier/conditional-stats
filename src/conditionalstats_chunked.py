"""@package conditionalstats_chunked
Documentation for module conditionalstats_chunked.

Class DistributionChunked defines objects containing the statistical distribution of a 
single variable, choosing the bin type (linear, logarithmic, inverse-logarithmic,
...). It inherits from class Distribution to combine distributions computed on subsets of the data.
"""

from conditionalstats import *
from scipy.interpolate import interp1d

class DistributionChunked(Distribution):
    """Documentation for class DistributionChunked
    
    Inherited class from parent class Distribution.
    """
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< DistributionChunked object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if k in ['dist_chunks','chunks_to_ignore']:
                # show type
                out = out+'%s\n'%str(getattr(self,k).__class__)
            else:
                if len(str(getattr(self,k))) < 80:
                    # show value
                    out = out+'%s\n'%str(getattr(self,k))
                else:
                    # show type
                    out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    def __str__(self):
        """Override string function to print attributes
        """
# #         # method_names = []
# #         # str_out = '-- Attributes --'
# #         str_out = ''
# #         for a in dir(self):
# #             if '__' not in a:
# #                 if a in ['dist_chunks','chunks_to_ignore']:
# #                     a_str = str(getattr(self,a).__class__)
# #                 else:
# #                     a_str = str(getattr(self,a))
# #                 if 'method' not in a_str:
# #                     str_out = str_out+("%s : %s\n"%(a,a_str))
                
# #         #         else:
# #         #             method_names.append(a)
# #         # print('-- Methods --')
# #         # for m in method_names:
# #         #     print(m)
# #         return str_out

        out = '< DistributionChunked object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if k in ['dist_chunks','chunks_to_ignore']:
                # show type
                out = out+'%s\n'%str(getattr(self,k).__class__)
            else:
                if len(str(getattr(self,k))) < 80:
                    # show value
                    out = out+'%s\n'%str(getattr(self,k))
                else:
                    # show type
                    out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    ##-- Class constructor

    def __init__(self,name='',dist_chunks=None,chunks_to_ignore=[],bintype='linear',nbpd=10,nppb=4,nbins=50,nd=None,\
        fill_last_decade=False,distribution=None,overwrite=False):
        """Constructor for class DistributionChunked.
        Inherited arguments from class Distribution:
        - name: name of reference variable
        - bintype [linear, log, invlog]: bin structure,
        - nbins: number of bins used for all types of statistics. Default is 50.
        - nbpd: number of bins per log or invlog decade. Default is 10.
        - nppb: minimum number of data points per bin. Default is 4.
        - nd: maximum number of decades in invlogQ bin type. Default is 4
        New arguments:
        - dist_chunks: list of distributions
        - chunks_to_ignore: indices in list of distributions to ignore for combined calculation 
        """
        
        # inherit method
        Distribution.__init__(self,name,bintype,nbpd,nppb,nbins,nd,fill_last_decade,distribution,overwrite)

        # additional arguments
        self.dist_chunks = dist_chunks#.copy()
        self.chunks_to_ignore = chunks_to_ignore.copy()
        
        # store parameters of global distribution
        self.nchunks = len(self.dist_chunks)
        self.size_chunks = [self.dist_chunks[i_d].size for i_d in range(self.nchunks)]
        self.vmin_chunks = [self.dist_chunks[i_d].vmin for i_d in range(self.nchunks)]
        self.vmax_chunks = [self.dist_chunks[i_d].vmax for i_d in range(self.nchunks)]
        self.size = np.sum(np.delete(self.size_chunks,self.chunks_to_ignore))
        self.vmin = np.min(np.delete(self.vmin_chunks,self.chunks_to_ignore))
        self.vmax = np.max(np.delete(self.vmax_chunks,self.chunks_to_ignore))
        
        self.bin_locations_stored = np.full((self.nchunks,),False)
        self.bin_locations = dict()
        
        # copy constructor
        if distribution is not None: # then copy it in self
            for attr in distribution.__dict__.keys():
                setattr(self,attr,getattr(distribution,attr)) 

    
    
    def setVminVmax(self,vmin=None,vmax=None,\
        overwrite=False):

        """Compute and set minimum and maximum values
        Arguments:
        - sample: 1D numpy array of data values."""

        # Find minimum value
        if vmin is None:
            vmin = np.min(np.delete(self.vmin_chunks,self.chunks_to_ignore))
        # Find maximum value
        if vmax is None:
            vmax = np.max(np.delete(self.vmax_chunks,self.chunks_to_ignore))
            
        if self.vmin is None or overwrite:
            self.vmin = vmin
        if self.vmax is None or overwrite:
            self.vmax = vmax

    # def getInvLogRanks(self,out=False) ##-- unchanged
        
    # def getLinRanks(self): ##-- unchanged

    def computePercentilesFromRanks(self,ranks=None,out=False,store=True,verbose=False):
        """New"""

        if ranks is None:
            target_ranks = self.ranks
        else:
            target_ranks = ranks

        N_Q = len(target_ranks)
        target_nless = np.array(self.size*target_ranks/100,dtype=int)  # to adapt in class Distribution
        # subset dimension axis
        d_sub = 1

        ##-- initialize

        # initial guess
        Pe_i = np.full(N_Q,40.0)
        # current estimate
        Pe_c = Pe_i.copy()
        Pe_n = Pe_i.copy()
        # search bounds
        B1 = np.full(N_Q,self.vmin)
        B2 = np.full(N_Q,self.vmax)
        # threshold
        err_thresh = self.vmax/10000
        if err_thresh == 0:
            raise ValueError("Error threshold is 0: will loop indefinitely")
        err = np.absolute(B2-B1)
        
        ##-- iterate

        if verbose:
            print('error:')

        while np.any(err > err_thresh):
            

            if verbose:
                print(u"%2.2f\u00B1%2.2f"%(np.mean(err),np.std(err)))

            ##-- calculate position of current estimate

            nless = np.full((N_Q,self.nchunks),np.nan) 
            for i_d in range(self.nchunks):

                if i_d in self.chunks_to_ignore: # skip
                    nless[:,i_d] = 0
                    continue

                dist = self.dist_chunks[i_d]

                #- deal with precip values outside subset range
                # above max value
                select_Pabove = Pe_c > dist.vmax
                nless[select_Pabove,i_d] = self.size_chunks[i_d]
                # below min value
                select_Pbelow = Pe_c < dist.vmin
                nless[select_Pbelow,i_d] = 0
                # in between:
                select_valid = np.logical_not(np.logical_or(select_Pabove,select_Pbelow))

                #- interpolate percentile rank Q(Pe|S)
                # interpolation function
                perc_all = np.hstack([dist.vmin,dist.percentiles,dist.vmax])
                Q_all = np.hstack([0,dist.ranks,100])
                Q_interp = interp1d(perc_all,Q_all)
                # perform interpolation on vector
                Q_est = Q_interp(Pe_c[select_valid]) # ! only keep valid points
                # infer number of points below
                nless[select_valid,i_d] = np.array(self.size_chunks[i_d]*Q_est/100,dtype=int) # ! only assign to valid points

            nless_tot = np.sum(nless,axis=d_sub) # add in time direction

            ##-- update search bounds and get new estimate

            #- for percentiles that are smaller than target
            nless_smaller = nless_tot < target_nless
            # move lower bound up
            B1[nless_smaller] = Pe_c[nless_smaller]
            # new estimate is mean between current estimate and upper bound
            Pe_n[nless_smaller] = np.mean(np.array([Pe_c[nless_smaller],B2[nless_smaller]]),axis=0)

            #- for percentiles that are larger than target
            nless_larger = nless_tot > target_nless
            # move lower bound up
            B2[nless_larger] = Pe_c[nless_larger]
            # new estimate is mean between current estimate and upper bound
            Pe_n[nless_larger] = np.mean(np.array([B1[nless_larger],Pe_c[nless_larger]]),axis=0)

            #- percentiles found exactly equal to target, by chance
            nless_equal = nless_tot == target_nless
            B1[nless_equal] = Pe_c[nless_equal]
            B2[nless_equal] = Pe_c[nless_equal]
            Pe_n[nless_equal] = Pe_c[nless_equal]

            ##-- update error and current estimate
            err = np.absolute(B2-B1)
            Pe_c[:] = Pe_n

        # store result
        if store:
            self.percentiles = Pe_c

        if out:
            return Pe_c

    
    # RECODE
    def computePercentilesAndBinsFromRanks(self,crop=False,store=True,out=False,verbose=False):

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

        ##-- percentiles
        percentiles = self.computePercentilesFromRanks(out=True,store=False,verbose=verbose)

        ##-- bins
        # calculate center bins (not minimum edge nor maximum edge)
        bins = np.array([np.nan]*(self.nbins+1))
        bins[1:-1] = self.computePercentilesFromRanks(ranks=self.rank_edges[1:-1],out=True,store=False,verbose=verbose)

        if not crop:
            bins[0] = self.vmin
            bins[-1] = self.vmax

        if store:
            self.percentiles = percentiles
            self.bins = bins
        
        if out:
            return self.percentiles, self.bins

    # RECODE -- redundant method, actually
    def definePercentilesOnInvLogQ(self,sample):

        """Defines percentiles and histogram bins on inverse-logarithmic ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """
        
        # First compute invlog ranks including its edge values
        self.getInvLogRanks()
        # Then compute final stats
        self.computePercentilesAndBinsFromRanks() # keep crop=False to get manually-set bounds


    # RECODE 
    def definePercentilesOnLinQ(self,sample,vmin=None,vmax=None):

        """Define percentiles and histogram bins on linear ranks.
        Arguments:
            - sample: 1D numpy array of values
        Sets:
            - ranks
            - percentiles
            - bins
        """
        
        pass
#         ## RECODE THIS SECTION WITH CHUNKS

#         self.setVminVmax(sample=sample,vmin=vmin,vmax=vmax)
#         # Compute linear ranks
#         self.getLinRanks()
#         # Then compute final stats
#         self.computePercentilesAndBinsFromRanks(sample)
        
#         ## END RECODE THIS SECTION WITH CHUNKS

    # RECODE
    def defineLogBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define logarithmic bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - n_bins_per_decade: number of ranks/bins per logarithmic decade
            - vmin and vmax: extremum values
        Computes:
            - centers (corresponding percentiles, or bin centers)
            - breaks (histogram bin edges)"""

        pass
#         ## RECODE THIS SECTION WITH CHUNKS
            
#         self.setVminVmax(sample,vmin,vmax,minmode)
#         kmin = floor(log10(self.vmin))
#         kmax = ceil(log10(self.vmax))
#         self.bins = np.logspace(kmin,kmax,(kmax-kmin)*self.nbpd)
#         self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
#         self.nbins = self.percentiles.size

#         ## END RECODE THIS SECTION WITH CHUNKS
        
    # RECODE
    def defineLinearBins(self,sample,vmin=None,vmax=None,minmode='positive'):

        """Define linear bin centers and edges from sample values.
        Arguments:
            - sample: 1D numpy array of values
            - vmin and vmax: extremum values
        Computes:
            - percentiles (or bin centers)
            - bins (edges)
        """

        pass
#         ## RECODE THIS SECTION WITH CHUNKS
        
#         self.setVminVmax(sample,vmin,vmax,minmode)
#         self.bins = np.linspace(self.vmin,self.vmax,self.nbins+1)
#         self.percentiles = np.convolve(self.bins,[0.5,0.5],mode='valid')
        
#         ## END RECODE THIS SECTION WITH CHUNKS
        
#         assert(self.percentiles.size == self.nbins), "wrong number of bins: #(percentiles)=%d and #(bins)=%d"%(self.percentiles.size,self.nbins)


    # RECODE
    def computePercentileRanksFromBins(self,sample):

        """Computes percentile ranks corresponding to percentile values.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks: 1D numpy.ndarray"""
        
        pass
        # self.ranks = 100*np.array(list(map(lambda x:(sample < x).sum()/self.size, \
        #     self.percentiles)))
        
    # RECODE
    def ranksPercentilesAndBins(self,vmin=None,vmax=None,verbose=False):

        """Preliminary step to compute probability densities. Define 
        ranks, percentiles, bins from the sample values and binning structure.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles and bins"""

        # self.setSampleSize(sample)
        self.setVminVmax(vmin,vmax)

        if self.bintype == 'linear':

            # self.defineLinearBins(sample,vmin,vmax,minmode) # OLD
            # self.computePercentileRanksFromBins(sample) # OLD
            raise WrongArgument("ERROR: code the case linear")            

        elif self.bintype == 'log':

            # self.defineLogBins(sample,vmin,vmax,minmode) # OLD
            # self.computePercentileRanksFromBins(sample) # OLD
            raise WrongArgument("ERROR: code the case log")

        elif self.bintype == 'invlogQ':

            self.getInvLogRanks()
            self.computePercentilesAndBinsFromRanks(verbose=verbose)

        elif self.bintype == 'linQ':

            # self.definePercentilesOnLinQ(sample) # OLD
            raise WrongArgument("ERROR: code the case linQ")

        else:

            raise WrongArgument("ERROR: unknown bintype")

            computePercentilesAndBinsFromRanks
            
    # RECODE
    def computeDistribution(self,vmin=None,vmax=None,verbose=False):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample: 1D numpy array of values
        Computes:
            - ranks, percentiles, bins and probability densities"""

        # # pass
        # if not self.overwrite:
        #     pass

        # Compute ranks, bins and percentiles
        self.ranksPercentilesAndBins(vmin,vmax,verbose=verbose)

#         ## RECODE THIS SECTION WITH CHUNKS

#         # Compute probability density
#         density, _ = np.histogram(sample,bins=self.bins,density=True)
#         self.density = density
#         # Number fraction of points below chosen vmin
#         self.frac_below_vmin = np.sum(sample < self.vmin)/np.size(sample)
#         # Number fraction of points above chosen vmax
#         self.frac_above_vmax = np.sum(sample > self.vmax)/np.size(sample)

#         ## END RECODE THIS SECTION WITH CHUNKS
        
    
    
    # def indexOfRank(self,rank): ##-- unchanged

    # def rankID(self,rank): ##-- unchanged

    # def binIndex(self,percentile=None,rank=None): ##--unchanged

    # def formatDimensions(self,sample): ##-- unchanged 

    # RECODE
    def storeSamplePoints(self,sample,sizemax,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping or extremes 
        and fetch locations later
        """

        if np.all(self.bin_locations_stored) and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")

        # # print(sample.shape)
        # sample = self.formatDimensions(sample)
        # # print(sample.shape)

        # Else initalize and find bin locations
        self.bin_locations = [[] for _ in range(self.nbins)]
        self.bin_sample_size = [0 for _ in range(self.nbins)]

#         if method == 'random':

#             # Here, look at all points, in random order
            
#             indices = list(range(self.size))
#             np.random.shuffle(indices)

#             bins_full = []
#             for i_ind in range(len(indices)):

#                 i = indices[i_ind]

#                 # Find corresponding bin
#                 i_bin = self.binIndex(percentile=sample[i])

#                 # Store only if bin was found
#                 if i_bin is not None:

#                     # Keep count
#                     self.bin_sample_size[i_bin] += 1
#                     # Store only if there is still room in stored locations list
#                     if len(self.bin_locations[i_bin]) < sizemax:
#                         self.bin_locations[i_bin].append(i)
#                     elif i_bin not in bins_full:
#                         bins_full.append(i_bin)
#                         bins_full.sort()
#                         if verbose:
#                             print("%d bins are full (%d iterations)"%(len(bins_full),i_ind))
            
#         elif method == 'shuffle_mask':

#             if verbose: print('bin #: ',end='')
#             # compute mask for each bin, randomize and store 'sizemax' first indices
#             for i_bin in range(self.nbins):

#                 if verbose: print('%d..'%i_bin,end='')

#                 # compute mask
#                 mask = np.logical_and(sample.flatten() >= self.bins[i_bin],
#                             sample.flatten() < self.bins[i_bin+1])
#                 # get all indices
#                 ind_mask = np.where(mask)[0]
#                 # shuffle
#                 np.random.seed(int(round(time.time() * 1000)) % 1000)
#                 np.random.shuffle(ind_mask)
#                 # select 'sizemax' first elements
#                 self.bin_locations[i_bin] = ind_mask[:sizemax]
#                 # self.bin_sample_size[i_bin] = min(ind_mask.size,sizemax) # cap at sizemax
#                 self.bin_sample_size[i_bin] = ind_mask.size # count all points there


#             if verbose: print()

#         if verbose:
#             print()

#         # If reach this point, everything should have worked smoothly, so:
#         self.bin_locations_stored = True

    # RECODE
    def computeMean(self,sample,out=False):
        """Compute mean of input sample"""
  
        pass
#         result = np.mean(sample)
#         setattr(self,"mean",result)
        
#         if out:
#             return result
        
    # RECODE
    def computeIndividualPercentiles(self,sample,ranks,out=False):
        """Computes percentiles of input sample and store in object attribute"""

        pass
#         if isinstance(ranks,float) or isinstance(ranks,int):
#             ranks = [ranks]
        
#         result = []

#         for r in ranks:
#             # calculate percentile
#             p = np.percentile(sample,r)
#             result.append(p)
#             # save
#             setattr(self,"perc%2.0f"%r,p)

#         if out:
#             return result

    # RECODE
    def computeInvCDF(self,sample,out=False):
        """Calculate 1-CDF on inverse-logarithmic ranks: fraction of rain mass falling 
        above each percentile"""
        
        pass
#         self.invCDF = np.ones(self.nbins)*np.nan
#         sample_sum = np.nansum(sample)
#         for iQ in range(self.nbins):
#             rank = self.ranks[iQ]
#             perc = self.percentiles[iQ]
#             if not np.isnan(perc):
#                 self.invCDF[iQ] = np.nansum(sample[sample>perc])/sample_sum

#         if out:
#             return self.invCDF

    # RECODE
    def bootstrapPercentiles(self,sample,nd_resample=10,n_bootstrap=50):
        """Perform bootstrapping to evaluate the interquartile range around each
        percentile, for the ranks stored.

        Arguments:
        - sample: np array in Nt,Ny,Nx format
        - nd_resample: number of time indices to randomly select for resampling
        - n_boostrap: number of times to calculate the distribution
        """

        pass
    
#         sshape = sample.shape
#         d_time = 0

#         # calculate and store distribution n_bootstrap times
#         perc_list = []
#         for i_b in range(n_bootstrap):

#             # select days randomly
#             indices = list(range(sshape[d_time]))
#             np.random.shuffle(indices)
#             ind_times = indices[:nd_resample]
#             resample = np.take(sample,ind_times,axis=d_time)

#             # calculate percentiles on resample
#             perc, bins = self.computePercentilesAndBinsFromRanks(resample,
#                                             store=False,output=True)

#             perc_list.append(perc)

#         # combine distributions into statistics and save
#         perc_array = np.vstack(perc_list)
#         self.percentiles_sigma = np.std(perc_array,axis=0)
#         self.percentiles_P5 = np.percentile(perc_array,5,axis=0)
#         self.percentiles_Q1 = np.percentile(perc_array,25,axis=0)
#         self.percentiles_Q2 = np.percentile(perc_array,50,axis=0)
#         self.percentiles_Q3 = np.percentile(perc_array,75,axis=0)
#         self.percentiles_P95 = np.percentile(perc_array,95,axis=0)
        

    # RECODE
    def getCDF(self):
        """Compute the cumulative density function from the probability density,
        as: fraction of points below vmin + cumulative sum of density*bin_width
        Output is the probability of x < x(bin i), same size as bins (bin edges)"""

        pass
#         # array of bin widths
#         bin_width = np.diff(self.bins)
#         # CDF from density and bin width
#         cdf_base = np.cumsum(bin_width*self.density)
#         # readjust  to account for the fraction of points outside the range [vmin,vmax]
#         fmin = self.frac_below_vmin
#         fmax = self.frac_above_vmax
#         cdf = fmin + np.append(0,cdf_base*(1-fmin-fmax))
        
#         return cdf
    
    