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
import CWDM_interpolator_new  as CWDM
from classy import Class
try:
    import BCemu
except:
    raise Exception ("Please install the BCemu package from https://github.com/sambit-giri/BCemu!")
try:
    import DMemu
except:
    raise Exception ("Please install the DMemu package from https://github.com/jbucko/DMemu!")
import DMemu_devel as DMemu_devel



class pdm(Likelihood):

    def __init__(self, path, data, command_line):
        
        Likelihood.__init__(self, path, data, command_line)

        # construct necessary objects:
        photo.initiate(self, data)
        if self.verbose: self.time = time()

        # reads fiducial file, if it exists:
        self.fid_values_exist = photo.read_data(self)

        # load emulators
        self.k_emu, self.ypca1_dict, self.gpr_load = CWDM.read_files()
        self.OBD = DMemu.OBDemu()
        self.TBD = DMemu.TBDemu()
        self.ethos = DMemu_devel.ETHOSn0emu()
        self.bfcemu = BCemu.BCM_3param(verbose=False)

        self.lcdm_pars = {}
        self.pk = np.zeros((self.lbin, self.nzmax), 'float64')
        self.k  = np.zeros((self.lbin, self.nzmax), 'float64')

        return

    def check_boundaries (self, cosmo, data):

        ##################################################
        # Check if BF or 1b parameters are out of bounds #
        ##################################################

        if data.use_BCemu:
            log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
            thej    = data.mcmc_parameters['thej']['current']    * data.mcmc_parameters['thej']['scale']
            deta    = data.mcmc_parameters['deta']['current']    * data.mcmc_parameters['deta']['scale']
            nu_Mc   = data.mcmc_parameters['nu_Mc']['current']   * data.mcmc_parameters['nu_Mc']['scale']
            nu_thej = data.mcmc_parameters['nu_thej']['current'] * data.mcmc_parameters['nu_thej']['scale']
            nu_deta = data.mcmc_parameters['nu_deta']['current'] * data.mcmc_parameters['nu_deta']['scale']

            fb = data.lcdm_pars['Omega_b'] / data.lcdm_pars['Omega_m']
            if fb < 0.1 or fb > 0.25:
                if self.verbose: print(" /!\ Skipping point because the baryon fraction is out of bounds!")
                return True

            if log10Mc / 3**nu_Mc < 11 or log10Mc / 3**nu_Mc > 15 \
                or thej / 3**nu_thej < 2 or thej / 3**nu_thej > 8 \
                or deta / 3**nu_deta < 0.05 or deta / 3**nu_deta > 0.4:
                if self.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
                return True

        if data.has_onebody:
            Gamma = data.onebody_pars['Gamma']
            f = data.onebody_pars['f_ddm_ini']
            if f < 0 or f > 1 or Gamma < 0 or Gamma > 0.0316:
                if self.verbose: print(" /!\ Skipping point because onebody-decay parameters are out of bounds!")
                return True

        return False

    def loglkl(self, cosmo, data):

        if self.check_boundaries(cosmo, data):
            return -1e10

        ##################
        # Initialisation #
        ##################

        if (self.verbose):              starting_time   = time()
        if self.print_individual_times: self.chunk_time = time()
        # relation between z and r (self.r also equals H(z)/c in 1/Mpc):
        self.r, self.dzdr = cosmo.z_of_r(self.z)

        ###################
        # LCDM CLASS call #
        ###################

        if data.lcdm_pars == self.lcdm_pars:
            # CLASS parameters have not changed, so we don't need to call it again
            # this is useful when you oversample
            pk = self.pk.copy()
            k  = self.k
        
        else:
            lcdm = Class()
            lcdm.set(data.lcdm_pars)
            lcdm.set({
                'output': 'mPk',
                'non linear': 'halofit',
                'nonlinear_min_k_max': self.kmax,
                'z_max_pk': self.z[-1],
                'P_k_max_1/Mpc': self.kmax
            })
            lcdm.compute(['fourier'])

            if self.print_individual_times:
                print("LCDM CLASS call:", time() - self.chunk_time, "s")
                self.chunk_time = time()

            ######################
            # Get power spectrum #
            ######################
            # Get power spectrum P(k=(l+1/2)/r,z) from cosmological module. Note that [P(k)] = Mpc^3

            pk = np.zeros((self.lbin, self.nzmax), 'float64')
            k = (self.l_high[:,None]+0.5)/self.r
            # ^ has units 1/Mpc

            cosmo_pk, cosmo_k, cosmo_z = lcdm.get_pk_and_k_and_z()
            pk_interpolator = RectBivariateSpline(cosmo_k,cosmo_z[::-1],np.flip(cosmo_pk,axis=1))
            # interpolate P(k) from those points where CLASS explicitly computed it
            for index_z in range(self.nzmax):
                ls_to_probe = np.where((k[:,index_z]>self.kmin) & (k[:,index_z]<self.kmax))[0]
                # ^ indexes of all values of k where P(k) won't be 0
                pk[ls_to_probe,index_z] = pk_interpolator (k[ls_to_probe,index_z], self.z[index_z])[:,0]

            self.lcdm_pars = data.lcdm_pars
            self.pk = pk.copy()
            self.k  = k

            if (self.print_individual_times):
                print("get power spectrum:", time() - self.chunk_time, "s")

        #################
        # PDM emulators #
        #################

        if data.has_cwdm:
            pk *= CWDM.CWDM_suppression (
                np.minimum(k / cosmo.h(), 20),
                # k must be converted back to units h/Mpc
                self.z,
                # k and z contain all ks and zs for all ls
                data.cwdm_pars['m_wdm']/1000.,
                # M_wdm needs to be parsed as keV
                data.cwdm_pars['f_wdm'],
                self.k_emu, self.ypca1_dict, self.gpr_load
            )

        # all emulators except CWDM have reasonable behaviour for k>kmax, so we don't have to do anything about it

        if data.has_onebody:
            # classy get_pk_and_k_and_z is broken sometimes, so we cannot use it
            # use the commented lines instead once it has been fixed
            ls_and_zs = [ ([l for l in range(self.lbin) if k[l,index_z] >= self.kmin and k[l,index_z] <= self.kmax], index_z) for index_z in range(self.nzmax)]
            pk_pdm_l  = np.zeros((self.lbin, self.nzmax), 'float64')
            pk_lcdm_l = np.zeros((self.lbin, self.nzmax), 'float64')
            # try:
            #     pdm_pk,  pdm_k,  pdm_z  = cosmo.get_pk_and_k_and_z(nonlinear=False)
            # except:
            #     print(" /!\ Severe error in CLASS get_pk_and_k_and_z, skipping point...")
            #     return -1e10
            lcdm_pk, lcdm_k, lcdm_z = lcdm.get_pk_and_k_and_z(nonlinear=False)
            # pk_interpolator_pdm_l  = RectBivariateSpline(pdm_k,  pdm_z  [::-1],np.flip(pdm_pk,  axis=1))
            pk_interpolator_lcdm_l = RectBivariateSpline(lcdm_k, lcdm_z [::-1],np.flip(lcdm_pk, axis=1))

            # interpolate P(k) from those points where CLASS explicitly computed it
            for index_z, z in enumerate(self.z):
                # pk_pdm_l   [ls_and_zs[index_z]] = pk_interpolator_pdm_l   (k[ls_and_zs[index_z]], z)[:,0]
                for index_l in ls_and_zs[index_z][0]:
                    pk_pdm_l[index_l,index_z] = cosmo.pk_lin(k[index_l,index_z], z)
                pk_lcdm_l  [ls_and_zs[index_z]] = pk_interpolator_lcdm_l  (k[ls_and_zs[index_z]], z)[:,0]
                pk         [ls_and_zs[index_z]] *= pk_pdm_l [ls_and_zs[index_z]] / pk_lcdm_l [ls_and_zs[index_z]] \
                    * self.OBD.predict(
                        k[ls_and_zs[index_z]]/cosmo.h(),
                        float(z),
                        data.onebody_pars['Gamma'],
                        data.onebody_pars['f_ddm_ini'],
                        cosmo.Omega_b(),
                        cosmo.Omega_m(),
                        cosmo.h()
                    )

        if data.has_twobody:
            ls_and_zs = [ ([l for l in range(self.lbin) if k[l,index_z] >= self.kmin and k[l,index_z] <= self.kmax], index_z) for index_z in range(self.nzmax)]
            for index_z, z in enumerate(self.z):
                pk[ls_and_zs[index_z]] *= self.TBD.predict(
                    k[ls_and_zs[index_z]]/cosmo.h(),
                    float(z),
                    data.twobody_pars['f_2b_ddm_0'],
                    data.twobody_pars['velocity_kick'],
                    data.twobody_pars['Gamma']
                )
        
        if data.has_ethos:
            ls_and_zs = [ ([l for l in range(self.lbin) if k[l,index_z] >= self.kmin and k[l,index_z] <= self.kmax], index_z) for index_z in range(self.nzmax)]
            for index_z, z in enumerate(self.z):
                pk[ls_and_zs[index_z]] *= self.ethos.predict(
                    k[ls_and_zs[index_z]]/cosmo.h(),
                    float(z),
                    data.ethos_pars['log10a_dark'],
                    data.ethos_pars['xi_idr']
                )

        if self.print_individual_times:
            print("PDM emulator call:", time() - self.chunk_time, "s")
            self.chunk_time = time()

        ###############################
        # BCemu baryonification model #
        ###############################

        if data.use_BCemu:
            # baryonic feedback modifications are only applied to k>kmin_bfc
            # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
            # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
            # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
            # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

            if not data.BCemu_affects_GC and self.probe_GC:
                pk_no_bf = pk.copy()
                pk_sqrt_bf = pk.copy()

            kmin_bfc = 0.035 * cosmo.h()
            kmax_bfc = 12.5  * cosmo.h()
            k_bfc = np.logspace(np.log10(max(kmin_bfc, self.kmin)), np.log10(min(kmax_bfc, self.kmax)), self.BCemu_k_bins)
            # ^ all have units 1/Mpc
            z_bfc = np.linspace(self.z[0], min(2, self.z[-1]), self.BCemu_z_bins)
            BFC = np.zeros((self.BCemu_k_bins, self.BCemu_z_bins))

            for index_z, z in enumerate(z_bfc):
                BFC[:,index_z] = self.bfcemu.get_boost(
                    z,
                    {
                        'log10Mc':  data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale'],
                        'thej':     data.mcmc_parameters['thej']['current']    * data.mcmc_parameters['thej']['scale'],
                        'deta':     data.mcmc_parameters['deta']['current']    * data.mcmc_parameters['deta']['scale'],
                        'nu_Mc':    data.mcmc_parameters['nu_Mc']['current']   * data.mcmc_parameters['nu_Mc']['scale'],
                        'nu_thej':  data.mcmc_parameters['nu_thej']['current'] * data.mcmc_parameters['nu_thej']['scale'],
                        'nu_deta':  data.mcmc_parameters['nu_deta']['current'] * data.mcmc_parameters['nu_deta']['scale']
                    },
                    k_bfc/cosmo.h(),
                    # convert to h/Mpc
                    cosmo.Omega_b()/cosmo.Omega_m())

            BFC_interpolator = RectBivariateSpline(k_bfc, z_bfc, BFC)
            for index_z, z in enumerate(self.z):
                ls_to_probe_bfc = np.where((k[:,index_z]>kmin_bfc) & (k[:,index_z]<self.kmax))[0]
                pk[ls_to_probe_bfc, index_z] *= BFC_interpolator(np.minimum(k[ls_to_probe_bfc, index_z], kmax_bfc), min(z, 2))[:,0]
                if not data.BCemu_affects_GC and self.probe_GC:
                    pk_sqrt_bf[ls_to_probe_bfc, index_z] *= BFC_interpolator(np.minimum(k[ls_to_probe_bfc, index_z], kmax_bfc), min(z, 2))[:,0]**0.5

        if self.print_individual_times:
            print("BCemu call:", time() - self.chunk_time, "s")
            self.chunk_time = time()

        ##############################################
        # Window functions W_L(z,bin) and W_G(z,bin) #
        ##############################################
        # in units of [W] = 1/Mpc

        if self.probe_WL:
            W_L = photo.window_functions_WL (self, cosmo, lcdm, data, k)
            # use LCDM for D(z)
        if self.probe_GC:
            W_G = photo.window_functions_GC (self, data)

        #######################
        # Write fiducial file #
        #######################
        # if it does not exist yet. Remember to use -f0 for this.

        if not self.fid_values_exist:
            legacy.euclid_photometric_z (self, cosmo, data, pk,
                pk_no_bf   if data.use_BCemu and not data.BCemu_affects_GC and self.probe_GC else None,
                pk_sqrt_bf if data.use_BCemu and not data.BCemu_affects_GC and self.probe_GC else None)
            return 1j

        ##################
        # Calculate chi2 #
        ##################

        try:
            if   self.probe_WL and not self.probe_GC: chi2 = c.chi2_WL (self, W_L, pk)
            elif self.probe_GC and not self.probe_WL: chi2 = c.chi2_GC (self, W_G, pk)
            elif self.probe_GC and     self.probe_WL: chi2 = c.chi2_xc (self, W_L, W_G, pk,
                pk_no_bf   if data.use_BCemu and not data.BCemu_affects_GC else pk,
                pk_sqrt_bf if data.use_BCemu and not data.BCemu_affects_GC else pk)
            else: raise Exception("No probe selected!")
            # chi2 = legacy.euclid_photometric_z (self, cosmo, data, pk, 
            #         pk_no_bf   if data.use_BCemu and not data.BCemu_affects_GC else pk,
            #         pk_sqrt_bf if data.use_BCemu and not data.BCemu_affects_GC else pk)
            # ^ use this line to let the old likelihood calculate chi2 for comparison
            
            if data.lcdm_pars != self.lcdm_pars: lcdm.struct_cleanup()

            if self.verbose:
                print("Euclid photometric chi2 =", chi2, "\t\telapsed time:", round(time() - self.time, 6), "s  \t(likelihood:", round(time() - starting_time, 6), "s)")
                self.time = time()
        except:
            return -1e10

        return -chi2/2.
