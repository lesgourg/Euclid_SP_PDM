########################################################
# Euclid photometric likelihood
########################################################
# written by Maike Doerenkamp in 2020
# following the recipe of 1910.09237 (Euclid preparation: VII. Forecast validation for Euclid
# cosmological probes)
# (adapted from euclid_lensing likelihood)
# edited by Lena Rathmann in 2021


from montepython.likelihood_class import Likelihood
import io_mp

import scipy.integrate
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy import interpolate as itp
from scipy.interpolate import UnivariateSpline,interp1d
import os
import numpy as np
import math
import warnings
import time
from scipy.special import erf

try:
    xrange
except NameError:
    xrange = range


class euclid_photometric_z(Likelihood):

    def __init__(self, path, data, command_line):
        self.debug_save  = True
        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_max_h_by_Mpc})


        # Compute non-linear power spectrum if requested
        if (self.use_halofit):
            self.need_cosmo_arguments(data, {'non linear':'halofit'})
            #self.need_cosmo_arguments(data, {'non linear':'HMcode'})


        # Define array of l values, evenly spaced in logscale

        if self.lmax_WL > self.lmax_GC:
            self.l_WL = np.logspace(np.log10(self.lmin), np.log10(self.lmax_WL), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_WL >= self.lmax_GC)[0])
            self.l_GC = self.l_WL[:self.idx_lmax+1]
            self.l_XC = self.l_WL[:self.idx_lmax+1]
            self.l_array = 'WL'
        else:
            self.l_GC = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_GC >= self.lmax_WL)[0])
            self.l_WL = self.l_GC[:self.idx_lmax+1]
            self.l_XC = self.l_GC[:self.idx_lmax+1]
            self.l_array = 'GC'
        #print('l array WL: ', self.l_WL)
        #print('l array GC: ', self.l_GC)
        if self.debug_save :
            np.savetxt('ls.txt',self.l_GC)
        ########################################################
        # Find distribution of n(z) in each bin
        ########################################################

        # Create the array that will contain the z boundaries for each bin.

        self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, self.zmax])

        # Fill array of discrete z values
        self.z = np.linspace(self.zmin, self.zmax, num=self.nzmax)

        # Fill distribution for each bin (convolving with photo_z distribution)
        # n_i = int n(z) dz over bin
        self.eta_z = np.zeros((self.nzmax, self.nbin), 'float64')
        self.photoerror_z = np.zeros((self.nzmax, self.nbin), 'float64')
        for Bin in xrange(self.nbin):
            for nz in xrange(self.nzmax):
                z = self.z[nz]
                self.photoerror_z[nz,Bin] = self.photo_z_distribution(z,Bin+1)
                self.eta_z[nz, Bin] = self.photoerror_z[nz,Bin] * self.galaxy_distribution(z)
        if self.debug_save : np.savetxt('./photoz.txt',self.photoerror_z) ## agrees
        if self.debug_save : np.savetxt('./unnorm_nofz.txt',self.eta_z) ## agrees
        # integrate eta(z) over z (in view of normalizing it to one)
        self.eta_norm = np.zeros(self.nbin, 'float64')
        #norm = np.array([trapz([self.photo_z_distribution(z1, i+1) for z1 in zint],dx=dz) for i in range(self.nbin)])
        for Bin in range(self.nbin):
            #self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],dx=self.zmax/self.nzmax)
            self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],self.z[:])

        if self.debug_save : np.savetxt('./n.txt',self.eta_z)
        # the normalised galaxy distribution per bin (dimensionless)
        #print('eta_z: ', self.eta_z)
        # the number density of galaxies per bin in inv sr
        self.n_bar = self.gal_per_sqarcmn * (60.*180./np.pi)**2
        self.n_bar /= self.nbin

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            self.bias = np.zeros(self.nbin)
            self.bias_names = []
            for ibin in xrange(self.nbin):
                self.bias_names.append('bias_'+str(ibin+1))
            self.nuisance += self.bias_names

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            self.nuisance += ['aIA', 'etaIA', 'betaIA']



        ###########
        # Read data
        ###########

        # If the file exists, initialize the fiducial values
        # It has been stored flat, so we use the reshape function to put it in
        # the right shape.

        if self.lmax_WL > self.lmax_GC:
            ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            l_jump = self.lmax_GC - self.lmin +1
            ells_GC = ells_WL[:l_jump]
        else:
            ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            l_jump = self.lmax_WL - self.lmin +1
            ells_WL = ells_GC[:l_jump]
        self.fid_values_exist = False
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            if 'WL' in self.probe:
                self.Cov_observ = np.zeros((len(ells_WL), self.nbin, self.nbin), 'float64')
            if 'GCph' in self.probe:
                self.Cov_observ = np.zeros((len(ells_GC), self.nbin, self.nbin), 'float64')
            if 'WL_GCph_XC' in self.probe:
                self.Cov_observ = np.zeros((l_jump, 2*self.nbin, 2*self.nbin), 'float64')
                if self.lmax_WL > self.lmax_GC:
                    l_high = len(ells_WL)-l_jump
                else:
                    l_high = len(ells_GC)-l_jump
                self.Cov_observ_high = np.zeros(((l_high), self.nbin, self.nbin), 'float64')
            with open(fid_file_path, 'r') as fid_file:
                line = fid_file.readline()
                while line.find('#') != -1:
                    line = fid_file.readline()
                while (line.find('\n') != -1 and len(line) == 1):
                    line = fid_file.readline()
                if 'WL' in self.probe:
                    for Bin1 in xrange(self.nbin):
                        for Bin2 in xrange(self.nbin):
                            for nl in xrange(len(ells_WL)):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                if 'GCph' in self.probe:
                    for Bin1 in xrange(self.nbin):
                        for Bin2 in xrange(self.nbin):
                            for nl in xrange(len(ells_GC)):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                if 'WL_GCph_XC' in self.probe:
                    for Bin1 in xrange(2*self.nbin):
                        for Bin2 in xrange(2*self.nbin):
                            for nl in xrange(l_jump):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                    for Bin1 in xrange(self.nbin):
                        for Bin2 in xrange(self.nbin):
                            for nl in xrange(l_high):
                                self.Cov_observ_high[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()

        return



    def galaxy_distribution(self, z):
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

        term1 =cb*f_out*erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))
        term2 =-cb*f_out*erf((0.707107*(z-z0-c0*self.z_bin_edge[bin]))/(sigma_0*(1+z)))
        term3 =c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin]))/(sigma_b*(1+z)))
        return (term1+term2+term3+term4)/(2*c0*cb)


    def loglkl(self, cosmo, data):

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')

        self.r, self.dzdr = cosmo.z_of_r(self.z)

        # H(z)/c in 1/Mpc
        H_z = self.dzdr
        # (H_0 /c) in 1/Mpc
        H0 = cosmo.h()/2997.92458


        kmin_in_inv_Mpc = self.k_min_h_by_Mpc * cosmo.h()
        kmax_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            # constant bias in each zbin, marginalise
            self.bias = np.zeros((self.nbin),'float64')
            if self.bias_model == 'binned_constant' :
                for ibin in xrange(self.nbin):
                    self.bias[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']

            elif self.bias_model == 'binned' :
                biaspars = dict()
                for ibin in xrange(self.nbin):
                    biaspars['b'+str(ibin+1)] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']
                brang = range(1,len(self.z_bin_edge))
                last_bin_num = brang[-1]
                def binbis(zz):
                    lowi = np.where( self.z_bin_edge <= zz )[0][-1]
                    if zz >= self.z_bin_edge[-1] and lowi == last_bin_num:
                        bii = biaspars['b'+str(last_bin_num)]
                    else:
                        bii = biaspars['b'+str(lowi+1)]
                    return bii
                vbinbis = np.vectorize(binbis)
                self.biasfunc = vbinbis

        ##############################################
        # Window functions W_L(z,bin) and W_G(z.bin) #
        ##############################################
        # in units of [W] = 1/Mpc

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            # Compute window function W_gamma(z) of lensing (without intrinsic alignement) for each bin:

            # - initialize all values to zero
            W_gamma = np.zeros((self.nzmax, self.nbin), 'float64')
            # - loop over bins
            for Bin in xrange(self.nbin):
                # - loop over z
                #   (the last value W_gamma[nzmax-1, Bin] is null by construction,
                #   so we can stop the loop at nz=nzmax-2)
                for nz in xrange(self.nzmax-1):
                    # - integrand defined for this z=z[nz]
                    integrand = self.eta_z[nz:, Bin]*(self.r[nz:]-self.r[nz])/self.r[nz:]
                    # - integral from z=z[nz] till z_max=z[nzmax-1]
                    W_gamma[nz, Bin] = trapz(integrand, self.z[nz:])
                    # - window function W_gamma(z) for the bin i=Bin
                    W_gamma[nz, Bin] *= 3./2.*H0**2. *cosmo.Omega_m()*self.r[nz]*(1.+self.z[nz])

            # Compute contribution from IA (Intrinsic Alignement)

            # - compute window function W_IA
            W_IA = self.eta_z *H_z[:,np.newaxis]

            # - IA contribution depends on a few parameters assigned here
            C_IA = 0.0134
            A_IA = data.mcmc_parameters['aIA']['current']*(data.mcmc_parameters['aIA']['scale'])
            eta_IA = data.mcmc_parameters['etaIA']['current']*(data.mcmc_parameters['etaIA']['scale'])
            beta_IA = data.mcmc_parameters['betaIA']['current']*(data.mcmc_parameters['betaIA']['scale'])

            # - read file for values of <L>/L*(z) (makes steps in z of 0.01)
            lum_file = open(os.path.join(self.data_directory,'scaledmeanlum_E2Sa.dat'), 'r')
            content = lum_file.readlines()
            zlum = np.zeros((len(content)))
            lum = np.zeros((len(content)))
            for index in xrange(len(content)):
                line = content[index]
                zlum[index] = line.split()[0]
                lum[index] = line.split()[1]
            # - create function lum(z)
            #   changed on 8.09: use linear interpolation like in CosmicFish
            #lum_func = UnivariateSpline(zlum, lum)
            lum_func = interp1d(zlum, lum,kind='linear')

            # - compute functions F_IA(z) and D(z)
            F_IA = (1.+self.z)**eta_IA * (lum_func(self.z))**beta_IA
            D_z = np.zeros((self.nzmax), 'float64')
            for index_z in xrange(self.nzmax):
                D_z[index_z] = cosmo.scale_independent_growth_factor(self.z[index_z])
                # note: for even better agreement with CosmicFish, could use
                # the scale dependent growth factor evaluated at k=0.0001 1/Mpc

            # - total lensing window function W_L(z,z_bin) including IA contribution
            W_L = np.zeros((self.nzmax, self.nbin), 'float64')
            W_L = W_gamma - A_IA*C_IA*cosmo.Omega_m()*F_IA[:,np.newaxis]/D_z[:,np.newaxis] *W_IA

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            # Compute window function W_G(z) of galaxy clustering for each bin:

            # - case where there is one constant bias value b_i for each bin i
            if self.bias_model == 'binned_constant' :
                W_G = np.zeros((self.nzmax, self.nbin), 'float64')
                W_G = self.bias[np.newaxis,:] * H_z[:,np.newaxis] * self.eta_z
            # - case where the bias is a single function b(z) for all bins
            if self.bias_model == 'binned' :
                W_G = np.zeros((self.nzmax, self.nbin), 'float64')
                W_G =  (H_z * self.biasfunc(self.z))[:,np.newaxis] * self.eta_z
            if self.debug_save :
                np.savetxt('./Hz.txt',H_z)
                np.savetxt('./windows.txt',W_G)
                np.savetxt('./z.txt',self.z)

        ################
        # Plot Windows #
        ################

        Plot_debug = False
        if Plot_debug == True:
            debug_file_path = os.path.join(
                self.data_directory, 'euclid_XC_W_z.dat')
            with open(debug_file_path, 'w') as debug_file:
                for Bin in xrange(self.nbin):
                    for nz in xrange(self.nzmax):
                        debug_file.write("%g  %.16g  %.16g %.16g\n" % (self.z[nz],W_L[nz, Bin],W_G[nz, Bin],W_gamma[nz, Bin]))
                    debug_file.write("\n")
                exit()

        ######################
        # Get power spectrum
        ######################
        # [P(k)] = Mpc^3

        # Get power spectrum P(k=(l+1/2)/r,z) from cosmological module

        if self.l_array == 'WL':
            l = self.l_WL
        if self.l_array == 'GC':
            l = self.l_GC
        k =(l[:,None]+0.5)/self.r

        pk = np.zeros((self.lbin, self.nzmax), 'float64')
        index_pknn = np.array(np.where((k> kmin_in_inv_Mpc) & (k<kmax_in_inv_Mpc))).transpose()
        for index_l, index_z in index_pknn:
            pk[index_l, index_z] = cosmo.pk(k[index_l, index_z], self.z[index_z])

        #############
        # Plot P(k) #
        #############

        Plot_debug = False
        if Plot_debug == True:
            debug_file_path = os.path.join(
                self.data_directory, 'euclid_xc_Pk.npy')
            with open(debug_file_path, 'w') as debug_file:
                np.save(debug_file, pk)
                np.save(debug_file, k)
                np.save(debug_file, z_kl)

        ##########
        # Noise
        ##########
        # dimensionless

        self.noise = {
           'LL': self.rms_shear**2./self.n_bar,
           'LG': 0.,
           'GL': 0.,
           'GG': 1./self.n_bar}

        #print('noise: ', self.noise)

        ##############
        # Calc Cl
        ##############
        # dimensionless
        # compute the LL component

        #print('pk: ', pk[1,1])
        #print('k: ', k[1,1])

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_LL = np.zeros((len(self.l_WL),self.nbin,self.nbin),'float64')
            Cl_LL_int = np.zeros((len(self.l_WL),self.nbin,self.nbin,self.nzmax),'float64')
            for nl in xrange(len(self.l_WL)):
                for Bin1 in xrange(self.nbin):
                    for Bin2 in xrange(Bin1,self.nbin):
                        Cl_LL_int[nl,Bin1,Bin2,:] = W_L[:, Bin1] * W_L[:, Bin2] * pk[nl,:] / H_z[:] / self.r[:] / self.r[:]
                        Cl_LL[nl,Bin1,Bin2] = trapz(Cl_LL_int[nl,Bin1,Bin2,:], self.z[:])

                        if Bin1==Bin2:
                            # add noise to diag elements
                            Cl_LL[nl,Bin1,Bin2] += self.noise['LL']
                        else:
                            # use symmetry of non-diag elems
                            Cl_LL[nl,Bin2,Bin1] = Cl_LL[nl,Bin1,Bin2]
        #print('Cl_LL: ', Cl_LL[1,2,3])

        # compute the GG component
        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_GG = np.zeros((len(self.l_GC),self.nbin,self.nbin),'float64')
            Cl_GG_int = np.zeros((len(self.l_GC),self.nbin,self.nbin,self.nzmax),'float64')
            for nl in xrange(len(self.l_GC)):
                for Bin1 in xrange(self.nbin):
                    for Bin2 in xrange(Bin1,self.nbin):
                        Cl_GG_int[nl,Bin1,Bin2,:] = W_G[:, Bin1] * W_G[:, Bin2] * pk[nl,:] / H_z[:] / self.r[:] / self.r[:]
                        Cl_GG[nl,Bin1,Bin2] = trapz(Cl_GG_int[nl,Bin1,Bin2,:], self.z[:])

                        if Bin1==Bin2:
                            # add noise to diag elems
                            Cl_GG[nl,Bin1,Bin2] += self.noise['GG']
                        else:
                            # use symmetry
                            Cl_GG[nl,Bin2,Bin1] = Cl_GG[nl,Bin1,Bin2]
        #print('Cl_GG: ', Cl_GG[1,2,3])

        # compute the GL component
        if 'WL_GCph_XC' in self.probe:
            Cl_LG = np.zeros((len(self.l_XC), self.nbin, self.nbin), 'float64')
            Cl_LG_int = np.zeros((len(self.l_XC), self.nbin, self.nbin, self.nzmax), 'float64')
            Cl_GL = np.zeros((len(self.l_XC), self.nbin, self.nbin), 'float64')
            for nl in xrange(len(self.l_XC)):
                for Bin1 in xrange(self.nbin):
                    for Bin2 in xrange(self.nbin):
                        # no symmetry of non-diag elems
                        Cl_LG_int[nl,Bin1,Bin2,:] = W_L[:, Bin1] * W_G[:, Bin2]  * pk[nl,:] / H_z[:] / self.r[:] / self.r[:]
                        Cl_LG[nl,Bin1,Bin2] = trapz(Cl_LG_int[nl,Bin1,Bin2,:], self.z[:])

                        # symmetry of LG and GL
                        Cl_GL[nl,Bin2,Bin1] = Cl_LG[nl,Bin1,Bin2]

                        if Bin1==Bin2:
                            Cl_LG[nl,Bin1,Bin2] += self.noise['LG']
                            Cl_GL[nl,Bin1,Bin2] += self.noise['GL']

        if self.debug_save :
            np.save('./clgg',Cl_GG)
        #print('Cl_LG: ', Cl_LG[1,2,3])

        ######################
        # Plot Cl's          #
        ######################

        Plot_debug = False
        if Plot_debug == True:
            Bin = 9
            debug_file_path = os.path.join(
                self.data_directory, 'z_Cl_'+str(Bin)+'.dat')
            with open(debug_file_path, 'w') as debug_file:
                for nl in xrange(len(self.l_XC)):
                    debug_file.write("%g  %.16g  %.16g  %.16g  %.16g\n" % (l[nl],Cl_LL[nl,Bin,Bin],Cl_GG[nl,Bin,Bin],Cl_LG[nl,Bin,Bin],Cl_GL[nl,Bin,Bin]))
            print("Printed Cl's")
            exit()

        Plot_debug = False
        if Plot_debug == True:
            if 'WL' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_WLz_Cl_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LL)
                debug_file_path = os.path.join(self.data_directory, 'euclid_WLz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_WL)
            if 'WL_GCph_XC' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_LL_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LL)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_GG_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_GG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_LG_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_XC)
            if 'GCph' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_GCz_Cl_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_GG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_GCz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_GC)
            exit()


        ########################
        # Spline Cl
        ########################
        # Find C(l) for every integer l

        # Spline the Cls along l
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            spline_LL = np.empty((self.nbin, self.nbin),dtype=(list,3))
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(self.nbin):
                    spline_LL[Bin1,Bin2] = list(itp.splrep(
                        self.l_WL[:], Cl_LL[:,Bin1,Bin2]))

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            spline_GG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(self.nbin):
                    spline_GG[Bin1,Bin2] = list(itp.splrep(
                        self.l_GC[:], Cl_GG[:,Bin1,Bin2]))

        if 'WL_GCph_XC' in self.probe:
            spline_LG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            spline_GL = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(self.nbin):
                    spline_LG[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_LG[:,Bin1,Bin2]))
                    spline_GL[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_GL[:,Bin1,Bin2]))

        # Create array of all integers of l
        if self.lmax_WL > self.lmax_GC:
            ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            l_jump = self.lmax_GC - self.lmin +1
            ells_GC = ells_WL[:l_jump]
        else:
            ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            l_jump = self.lmax_WL - self.lmin +1
            ells_WL = ells_GC[:l_jump]

        if 'WL_GCph_XC' in self.probe:

            Cov_theory = np.zeros((l_jump, 2*self.nbin, 2*self.nbin), 'float64')
            if self.lmax_WL > self.lmax_GC:
                Cov_theory_high = np.zeros(((len(ells_WL)-l_jump), self.nbin, self.nbin), 'float64')
            else:
                Cov_theory_high = np.zeros(((len(ells_GC)-l_jump), self.nbin, self.nbin), 'float64')
        elif 'WL' in self.probe:
            Cov_theory = np.zeros((len(ells_WL), self.nbin, self.nbin), 'float64')
        elif 'GCph' in self.probe:
            Cov_theory = np.zeros((len(ells_GC), self.nbin, self.nbin), 'float64')

        for Bin1 in xrange(self.nbin):
            for Bin2 in xrange(self.nbin):
                if 'WL_GCph_XC' in self.probe:
                    if self.lmax_WL > self.lmax_GC:
                        Cov_theory[:,Bin1,Bin2] = itp.splev(
                            ells_GC[:], spline_LL[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,Bin2] = itp.splev(
                            ells_GC[:], spline_GL[Bin1,Bin2])
                        Cov_theory[:,Bin1,self.nbin+Bin2] = itp.splev(
                            ells_GC[:], spline_LG[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,self.nbin+Bin2] = itp.splev(
                            ells_GC[:], spline_GG[Bin1,Bin2])

                        Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                            ells_WL[l_jump:], spline_LL[Bin1,Bin2])
                    else:
                        Cov_theory[:,Bin1,Bin2] = itp.splev(
                            ells_WL[:], spline_LL[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,Bin2] = itp.splev(
                            ells_WL[:], spline_GL[Bin1,Bin2])
                        Cov_theory[:,Bin1,self.nbin+Bin2] = itp.splev(
                            ells_WL[:], spline_LG[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,self.nbin+Bin2] = itp.splev(
                            ells_WL[:], spline_GG[Bin1,Bin2])

                        Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                            ells_GC[l_jump:], spline_LL[Bin1,Bin2])

                elif 'WL' in self.probe:
                    Cov_theory[:,Bin1,Bin2] = itp.splev(
                        ells_WL[:], spline_LL[Bin1,Bin2])

                elif 'GCph' in self.probe:
                    Cov_theory[:,Bin1,Bin2] = itp.splev(
                        ells_GC[:], spline_GG[Bin1,Bin2])

        #print('Cov_theory: ', Cov_theory[1,2,3])

        #######################
        # Create fiducial file
        #######################

        if self.fid_values_exist is False:
            # Store the values now, and exit.
            fid_file_path = os.path.join(
                self.data_directory, self.fiducial_file)
            with open(fid_file_path, 'w') as fid_file:
                fid_file.write('# Fiducial parameters')
                for key, value in data.mcmc_parameters.items():
                    fid_file.write(
                        ', %s = %.5g' % (key, value['current']*value['scale']))
                fid_file.write('\n')
                if 'WL' in self.probe or 'GCph' in self.probe:
                    for Bin1 in xrange(self.nbin):
                        for Bin2 in xrange(self.nbin):
                            for nl in xrange(len(Cov_theory[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
                if 'WL_GCph_XC' in self.probe:
                    for Bin1 in xrange(2*self.nbin):
                        for Bin2 in xrange(2*self.nbin):
                            for nl in xrange(len(Cov_theory[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
                    for Bin1 in xrange(self.nbin):
                        for Bin2 in xrange(self.nbin):
                            for nl in xrange(len(Cov_theory_high[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory_high[nl, Bin1, Bin2])
            print('\n')
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        ######################
        # Compute likelihood
        ######################
        # Define cov theory and observ on the whole integer range of ell values

        chi2 = 0.

        if 'WL_GCph_XC' in self.probe:
            if self.lmax_WL > self.lmax_GC:
                ells = ells_WL
            else:
                ells = ells_GC
            for index, ell in enumerate(ells):

                if ell<=self.lmax_XC:
                    det_theory = np.linalg.det(Cov_theory[index,:,:])
                    det_observ = np.linalg.det(self.Cov_observ[index,:,:])
                    if det_theory/det_observ <= 0. :
                        print('l, det_theory, det_observ, ratio', ell, det_theory, det_observ, det_theory/det_observ)

                    det_cross = 0.
                    for i in xrange(2*self.nbin):
                        newCov = np.copy(Cov_theory[index, :, :])
                        newCov[:, i] = self.Cov_observ[index, :, i]
                        det_cross += np.linalg.det(newCov)/det_theory
                    #if index==2:
                        #print('det_theory: ', det_theory)
                        #print('det_observ: ', det_observ)
                        #print('det_cross: ', det_cross)
                    chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - 2*self.nbin)

                else:
                    det_theory = np.linalg.det(Cov_theory_high[ell-self.lmax_XC-1,:,:])
                    det_observ = np.linalg.det(self.Cov_observ_high[ell-self.lmax_XC-1,:,:])
                    if det_theory/det_observ <= 0. :
                        print('l, det_theory, det_observ, ratio', ell, det_theory, det_observ, det_theory/det_observ)

                    det_cross = 0.
                    for i in xrange(self.nbin):
                        newCov = np.copy(Cov_theory_high[ell-self.lmax_XC-1, :, :])
                        newCov[:, i] = self.Cov_observ_high[ell-self.lmax_XC-1, :, i]
                        det_cross += np.linalg.det(newCov)/det_theory
                    #if index==2:
                        #print('det_theory: ', det_theory)
                        #print('det_observ: ', det_observ)
                        #print('det_cross: ', det_cross)

                    chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - self.nbin)


        elif 'WL' in self.probe:
            for index, ell in enumerate(ells_WL):
                det_theory = np.linalg.det(Cov_theory[index,:,:])
                det_observ = np.linalg.det(self.Cov_observ[index,:,:])

                det_cross = 0.
                for i in xrange(self.nbin):
                    newCov = np.copy(Cov_theory[index, :, :])
                    newCov[:, i] = self.Cov_observ[index, :, i]
                    det_cross += np.linalg.det(newCov)/det_theory

                chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - self.nbin)

        elif 'GCph' in self.probe:
            for index, ell in enumerate(ells_GC):
                det_theory = np.linalg.det(Cov_theory[index,:,:])
                det_observ = np.linalg.det(self.Cov_observ[index,:,:])

                det_cross = 0.
                for i in xrange(self.nbin):
                    newCov = np.copy(Cov_theory[index, :, :])
                    newCov[:, i] = self.Cov_observ[index, :, i]
                    det_cross += np.linalg.det(newCov)/det_theory

                chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - self.nbin)

        print("euclid photometric: chi2 = ",chi2)
        return -chi2/2.
