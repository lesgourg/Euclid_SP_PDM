import numpy as np
from scipy.special import erf
from scipy.integrate import trapz
from scipy.interpolate import interp1d, RectBivariateSpline
import os, sys
from time import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import euclid_photo_c_wrapper as c



######################################
# This code is called once per chain #
######################################



def initiate (self, data):

    self.kmin = self.kmin_in_inv_Mpc if hasattr(self, 'kmin_in_inv_Mpc') else self.kmin_in_h_by_Mpc * cosmo.h()
    self.kmax = self.kmax_in_inv_Mpc if hasattr(self, 'kmax_in_inv_Mpc') else self.kmax_in_h_by_Mpc * cosmo.h()
    # ^ have units 1/Mpc

    # Create the array that will contain the z boundaries for each bin.
    self.z_bin_edge = np.array(self.z_bin_edges)
    self.zmin = self.z_bin_edge[0]
    self.zmax = self.z_bin_edge[-1]
    self.nbin = len(self.z_bin_edge) - 1
    # Fill array of discrete z values
    self.z = np.linspace(self.zmin, self.zmax, num=self.nzmax)

    # Define arrays of l values, evenly spaced in logscale
    if self.probe_WL and not self.probe_GC:
        self.l_high = np.logspace(np.log10(self.lmin), np.log10(self.lmax_WL), num=self.lbin, endpoint=True)
    if self.probe_GC and not self.probe_WL:
        self.l_high = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
    else:
        self.l_high = np.logspace(np.log10(self.lmin), np.log10(max(self.lmax_WL, self.lmax_GC)), num=self.lbin, endpoint=True)
        self.l_low = self.l_high[:np.argwhere( self.l_high >= min(self.lmax_WL, self.lmax_GC) )[0,0] + 1]
        # ^ l values for the probe reaching higher ls and the probe reaching lower ls

    if data.has_onebody or self.scale_dependent_f:
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': self.kmax_in_inv_Mpc} if hasattr(self, 'kmax_in_inv_Mpc') else {'P_k_max_h/Mpc': self.kmax_in_h_by_Mpc})

    if not 'OMP_NUM_THREADS' in os.environ: print(" /!\ Set environment variable OMP_NUM_THREADS to enable multithreading in euclid_photometric_fast.\n")

    # Fill distribution for each bin (convolving with photo_z distribution)
    # n_i = int n(z) dz over bin
    self.eta_z = np.zeros((self.nzmax, self.nbin), 'float64')
    for Bin in range(self.nbin):
        self.eta_z[:, Bin] = photo_z_distribution(self, self.z, Bin+1)
    self.eta_z *= galaxy_distribution(self.z)[:,None]
    # integrate eta(z) over z (in view of normalizing it to one)
    for Bin in range(self.nbin):
        self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin], self.z)

    # the number density of galaxies per bin in inv sr
    self.n_bar = self.gal_per_sqarcmn * (60.*180./np.pi)**2 / self.nbin
    self.WL_noise = self.rms_shear**2./self.n_bar
    self.GC_noise = 1./self.n_bar

    if self.probe_GC:   self.nuisance += ['bias_1', 'bias_2', 'bias_3', 'bias_4', 'bias_5', 'bias_6', 'bias_7', 'bias_8', 'bias_9', 'bias_10']
    if self.probe_WL:   self.nuisance += ['aIA', 'etaIA', 'betaIA']
    if data.use_BCemu:  self.nuisance += ['log10Mc', 'thej', 'deta', 'nu_Mc', 'nu_thej', 'nu_deta']

    for nuisance in self.nuisance:
        assert(nuisance in data.mcmc_parameters), "Missing nuisance parameters in your input file! Could not find \'" + str(nuisance) + "\'."
    
    ####################################
    # Weak Lensing Luminosity Function #
    ####################################

    # - read file for values of <L>/L*(z) (makes steps in z of 0.01)
    lum_file = open(os.path.join(self.data_directory,'scaledmeanlum_E2Sa.dat'), 'r')
    content = lum_file.readlines()
    zlum = np.zeros((len(content)))
    lum  = np.zeros((len(content)))
    for index in range(len(content)):
        line = content[index]
        zlum[index] = line.split()[0]
        lum [index] = line.split()[1]
    # - create function lum(z)
    self.lum_func = interp1d(zlum, lum, kind='linear')



def read_data (self):
    fid_file_path = os.path.join(self.data_directory, self.fiducial_file_path)
    if os.path.exists(fid_file_path):
        loaded = np.load(fid_file_path)
        self.Cov_observ = loaded['Cov_observ']
        if self.probe_WL and self.probe_GC:
            self.Cov_observ_high = loaded['Cov_observ_high']
        return True
    else:
        print(" /!\ No fiducial file found, generating fiducial...")
        return False



def galaxy_distribution(z):
    """
    Galaxy distribution returns the function D(z) from the notes

    Modified by S. Clesse in March 2016 to add an optional form of n(z) motivated by ground based exp. (Van Waerbeke et al., 2013)
    See google doc document prepared by the Euclid IST - Splinter 2
    """
    zmean = 0.9
    z0 = zmean/np.sqrt(2)

    galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))

    return galaxy_dist



def photo_z_distribution(self, z, bin):
    """
    Photo z distribution

    z:      physical galaxy redshift
    zph:    measured galaxy redshift
    """

    c0, z0, sigma_0 = 1.0, 0.1, 0.05
    cb, zb, sigma_b = 1.0, 0.0, 0.05
    f_out = 0.1

    if bin == 0 or bin >= 11:
        return None

    term1 =    cb*f_out *erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))
    term2 =   -cb*f_out *erf((0.707107*(z-z0-c0*self.z_bin_edge[bin]    ))/(sigma_0*(1+z)))
    term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))
    term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin]    ))/(sigma_b*(1+z)))
    return (term1+term2+term3+term4)/(2*c0*cb)



#####################################
# This code is called at every step #
#####################################



def window_functions_WL(self, cosmo, lcdm, data, k):
    # Compute window function W_gamma(z) of lensing (without intrinsic alignement) for each bin:

    # Calculate W_gamma in C++. An explanation of the steps can be found in the original likelihood.
    W_gamma = c.W_gamma (self, cosmo)

    # 1b-ddm correction factor that accounts for the fact that rho_m is not proportional to a^{-3} anymore
    corr_factor_at_z = np.ones(self.nzmax)
    for i, zz in enumerate(self.z):
        corr_factor_at_z[i] = cosmo.Om_m(zz) * cosmo.Hubble(zz)**2 
    # print(corr_factor_at_z)
    W_gamma *= (corr_factor_at_z * (1+self.z)**-3 / cosmo.Om_m(0) / cosmo.Hubble(0)**2)[:,None]

    # Compute contribution from IA (Intrinsic Alignement)
    # - compute window function W_IA
    W_IA = self.eta_z * self.dzdr[:,None]

    # - IA contribution depends on a few parameters assigned here
    C_IA = 0.0134
    A_IA    = data.mcmc_parameters['aIA']['current']   *(data.mcmc_parameters['aIA']['scale'])
    eta_IA  = data.mcmc_parameters['etaIA']['current'] *(data.mcmc_parameters['etaIA']['scale'])
    beta_IA = data.mcmc_parameters['betaIA']['current']*(data.mcmc_parameters['betaIA']['scale'])

    # - compute functions F_IA(z) and D(z)
    F_IA = (1.+self.z)**eta_IA * (self.lum_func(self.z))**beta_IA

    # - total lensing window function W_L(z,z_bin) including IA contribution
    if not self.scale_dependent_f:
        # here, D(z) is calculated using an analytical approximation that is _only valid for LCDM_!
        D_z = np.zeros((self.nzmax), 'float64')
        for index_z, z in enumerate(self.z):
            D_z[index_z] = lcdm.scale_independent_growth_factor(z)
        W_L = W_gamma - A_IA*C_IA*cosmo.Omega_m()*F_IA[:,None]/D_z[:,None]*W_IA

    else:
        # calculates the scale-dependent D(k,z) from the _linear_ power spectrum. Useful for non-cold dark matter models,
        # neutrinos or modified gravity theories.
        # We should have a function for this in Classy (or even CLASS).
        D_z = np.ones((self.lbin,self.nzmax), 'float64')

        try:
            Pk_l_grid, k_grid, z_grid = cosmo.get_pk_and_k_and_z (nonlinear=False)
            Pk_l = RectBivariateSpline(z_grid[::-1], k_grid, (np.flip(Pk_l_grid, axis=1)).transpose())
            for index_z, z in enumerate(self.z):
                ls_to_probe = np.where((k[:,index_z]>self.kmin) & (k[:,index_z]<self.kmax))
                D_z[ls_to_probe,index_z] = np.sqrt(Pk_l(z, k[ls_to_probe,index_z]) / Pk_l(0, k[ls_to_probe,index_z]))
        except:
            for index_z, z in enumerate(self.z):
                ls_to_probe = np.where((k[:,index_z]>self.kmin) & (k[:,index_z]<self.kmax))
                for index_l in ls_to_probe[0]:
                    D_z[index_l,index_z] = np.sqrt( cosmo.pk_lin(k[index_l, index_z], z) / cosmo.pk_lin(k[index_l, index_z], 0) )

        W_L = W_gamma[None,:,:] - A_IA*C_IA*cosmo.Omega_m() * F_IA[None,:,None]/D_z[:,:,None] * W_IA[None,:,:]

    if not self.probe_GC and self.print_individual_times:
        print("calculate window functions:", time() - self.chunk_time, "s")
        self.chunk_time = time()

    return W_L



def window_functions_GC(self, data):
    # Compute window function W_G(z) of galaxy clustering for each bin:

    # - case where there is one constant bias value b_i for each bin i
    if self.bias_model == 'binned_constant':
        # constant bias in each zbin, marginalise
        bias = np.array([data.mcmc_parameters[self.nuisance[ibin]]['current']*data.mcmc_parameters[self.nuisance[ibin]]['scale'] for ibin in range(self.nbin)])
        W_G = bias[None,:] * self.dzdr[:,None] * self.eta_z

    # - case where the bias is a single function b(z) for all bins
    elif self.bias_model == 'binned':
        def binbis(zz):
            lowi = np.where(self.z_bin_edge <= zz)[0][-1]
            if zz >= self.zmax:
                bii = data.mcmc_parameters[self.nuisance[self.nbin-1]]['current']*data.mcmc_parameters[self.nuisance[self.nbin-1]]['scale']
            else:
                bii = data.mcmc_parameters[self.nuisance[lowi]]['current']*data.mcmc_parameters[self.nuisance[lowi]]['scale']
            return bii
        vbinbis = np.vectorize(binbis)(self.z)
        W_G = (self.dzdr * vbinbis)[:,None] * self.eta_z

    else:
        raise Exception("Chosen bias model not implemented!")

    if self.print_individual_times:
        print("calculate window functions:", time() - self.chunk_time, "s")
        self.chunk_time = time()

    return W_G
