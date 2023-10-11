#################################
# Euclid photometric likelihood #
#################################
# Originally written by Maike Doerenkamp in 2020 and edited by Lena Rathmann in 2021 and others
# later on. Follows the recipe of 1910.09237. Adapted from euclid_lensing likelihood.

# This version is a fork of the euclid_photometric_z likelihood which has been significantly
# sped up by using more efficient programming and rewriting large parts of the the code in C++.
# Before your first run, or if you modify the C++ part, you will have to (re-) compile it into
# the shared library ./c++/euclid_photo.so using ./c++/compile.sh (you may have to mark the .sh
# file as executable first).
# Different routines for the splines and matrix determinant calculations may result in small
# numerical deviations from the chi2 as calculated by the python implementation.
# If there are significant differences between euclid_photometric_z and this likelihood, trust
# euclid_photometric_z and not this.
# - Justus Schwagereit, 2023



import numpy as np
from scipy.interpolate import RectBivariateSpline
from time import time
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from montepython.likelihood_class import Likelihood
import euclid_photo_functions as photo
import euclid_photo_c_wrapper as c
import euclid_legacy_code     as legacy



class euclid_photometric_fast(Likelihood):

    def __init__(self, path, data, command_line):
        
        Likelihood.__init__(self, path, data, command_line)

        # construct necessary objects:
        photo.initiate(self, data)
        if self.verbose: self.time = time()

        # reads fiducial file, if it exists:
        self.fid_values_exist = photo.read_data(self)

        return

    def loglkl(self, cosmo, data):

        ##################
        # Initialisation #
        ##################

        if (self.verbose):              starting_time   = time()
        if self.print_individual_times: self.chunk_time = time()
        # relation between z and r (self.r also equals H(z)/c in 1/Mpc):
        self.r, self.dzdr = cosmo.z_of_r(self.z)

        ######################
        # Get power spectrum #
        ######################
        # Get power spectrum P(k=(l+1/2)/r,z) from cosmological module. Note that [P(k)] = Mpc^3

        pk = np.zeros((self.lbin, self.nzmax), 'float64')
        k = (self.l_high[:,None]+0.5)/self.r
        kmin = self.kmin_in_inv_Mpc if hasattr(self, 'kmin_in_inv_Mpc') else self.kmin_in_h_by_Mpc * cosmo.h()
        kmax = self.kmax_in_inv_Mpc if hasattr(self, 'kmax_in_inv_Mpc') else self.kmax_in_h_by_Mpc * cosmo.h()

        cosmo_pk, cosmo_k, cosmo_z = cosmo.get_pk_and_k_and_z()
        pk_interpolator = RectBivariateSpline(cosmo_k,cosmo_z[::-1],np.flip(cosmo_pk,axis=1))
        # interpolate P(k) from those points where CLASS explicitly computed it
        for index_z in range(self.nzmax):
            ls_to_probe = np.where((k[:,index_z]>kmin) & (k[:,index_z]<kmax))[0]
            # ^ indexes of all values of k where P(k) won't be 0
            pk[ls_to_probe,index_z] = pk_interpolator (k[ls_to_probe,index_z], self.z[index_z])[:,0]
            # for l in ls_to_probe:
            #     pk[l, index_z] = cosmo.pk(k[l, index_z], self.z[index_z])

        if (self.print_individual_times):
            print("get power spectrum:", time() - self.chunk_time, "s")

        ##############################################
        # Window functions W_L(z,bin) and W_G(z,bin) #
        ##############################################
        # in units of [W] = 1/Mpc

        if self.probe_WL:
            W_L = photo.window_functions_WL (self, cosmo, data, k, kmin, kmax)
        if self.probe_GC:
            W_G = photo.window_functions_GC (self, data)

        #######################
        # Write fiducial file #
        #######################
        # if it does not exist yet. Remember to use -f0 for this.

        if not self.fid_values_exist:
            legacy.euclid_photometric_z (self, cosmo, data, pk)
            return 1j

        ##################
        # Calculate chi2 #
        ##################

        if   self.probe_WL and not self.probe_GC: chi2 = c.chi2_WL (self, W_L, pk)
        elif self.probe_GC and not self.probe_WL: chi2 = c.chi2_GC (self, W_G, pk)
        elif self.probe_GC and     self.probe_WL: chi2 = c.chi2_xc (self, W_L, W_G, pk)
        else: raise Exception("No probe selected!")
        # chi2 = legacy.euclid_photometric_z (self, cosmo, data, pk)
        # ^ use this line to let the old likelihood calculate chi2 for comparison

        if self.verbose:
            print("Euclid photometric chi2 =", chi2, "\t\telapsed time:", round(time() - self.time, 6), "s  \t(likelihood:", round(time() - starting_time, 6), "s)")
            self.time = time()
        return -chi2/2.
