from scipy.integrate import trapz
from scipy import interpolate as itp
from scipy.interpolate import interp1d
import os
import numpy as np
import warnings



# This code has simply been copy-pasted from the original euclid_photometric_z
# likelihood and has not been improved in any way.
# It is used to generate the fiducial file.



def euclid_photometric_z (self, cosmo, data, pk, pk_no_bf=None, pk_sqrt_bf=None):
    if pk_no_bf is None: pk_no_bf = pk
    if pk_sqrt_bf is None: pk_sqrt_bf = pk

    if self.probe_WL and not self.probe_GC: self.probe = ['WL']
    elif self.probe_GC and not self.probe_WL: self.probe = ['GCph']
    elif self.probe_GC and     self.probe_WL: self.probe = ['WL_GCph_XC']
    else: raise Exception ("No probe selected!")
    idx_lmax = np.argwhere( self.l_high >= min(self.lmax_WL, self.lmax_GC) )[0,0]
    self.l_XC = self.l_high[:idx_lmax+1]
    if self.lmax_WL > self.lmax_GC:
        self.l_array = 'WL'
        self.l_WL = self.l_high
        self.l_GC = self.l_XC
    else:
        self.l_array = 'GC'
        self.l_GC = self.l_high
        self.l_WL = self.l_XC
    # One wants to obtain here the relation between z and r, this is done
    # by asking the cosmological module with the function z_of_r
    self.r = np.zeros(self.nzmax, 'float64')
    self.dzdr = np.zeros(self.nzmax, 'float64')

    self.r, self.dzdr = cosmo.z_of_r(self.z)

    # H(z)/c in 1/Mpc
    H_z = self.dzdr
    # (H_0 /c) in 1/Mpc
    H0 = cosmo.h()/2997.92458

    if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
        # constant bias in each zbin, marginalise
        self.bias = np.zeros(self.nbin)
        self.bias_names = []
        for ibin in range(self.nbin):
            self.bias_names.append('bias_'+str(ibin+1))
        self.nuisance += self.bias_names
        if self.bias_model == 'binned_constant' :
            for ibin in range(self.nbin):
                self.bias[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']

        elif self.bias_model == 'binned' :
            biaspars = dict()
            for ibin in range(self.nbin):
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
        for Bin in range(self.nbin):
            # - loop over z
            #   (the last value W_gamma[nzmax-1, Bin] is null by construction,
            #   so we can stop the loop at nz=nzmax-2)
            for nz in range(self.nzmax-1):
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
        for index in range(len(content)):
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
        for index_z in range(self.nzmax):
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

    ##########
    # Noise
    ##########
    # dimensionless

    self.noise = {
        'LL': self.rms_shear**2./self.n_bar,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./self.n_bar}

    ##############
    # Calc Cl
    ##############
    # dimensionless
    # compute the LL component

    if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
        Cl_LL = np.zeros((len(self.l_WL),self.nbin,self.nbin),'float64')
        Cl_LL_int = np.zeros((len(self.l_WL),self.nbin,self.nbin,self.nzmax),'float64')
        for nl in range(len(self.l_WL)):
            for Bin1 in range(self.nbin):
                for Bin2 in range(Bin1,self.nbin):
                    Cl_LL_int[nl,Bin1,Bin2,:] = W_L[:, Bin1] * W_L[:, Bin2] * pk[nl,:] / H_z[:] / self.r[:] / self.r[:]
                    Cl_LL[nl,Bin1,Bin2] = trapz(Cl_LL_int[nl,Bin1,Bin2,:], self.z[:])

                    if Bin1==Bin2:
                        # add noise to diag elements
                        Cl_LL[nl,Bin1,Bin2] += self.noise['LL']
                    else:
                        # use symmetry of non-diag elems
                        Cl_LL[nl,Bin2,Bin1] = Cl_LL[nl,Bin1,Bin2]

    # compute the GG component
    if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
        Cl_GG = np.zeros((len(self.l_GC),self.nbin,self.nbin),'float64')
        Cl_GG_int = np.zeros((len(self.l_GC),self.nbin,self.nbin,self.nzmax),'float64')
        for nl in range(len(self.l_GC)):
            for Bin1 in range(self.nbin):
                for Bin2 in range(Bin1,self.nbin):
                    Cl_GG_int[nl,Bin1,Bin2,:] = W_G[:, Bin1] * W_G[:, Bin2] * pk_no_bf[nl,:] / H_z[:] / self.r[:] / self.r[:]
                    Cl_GG[nl,Bin1,Bin2] = trapz(Cl_GG_int[nl,Bin1,Bin2,:], self.z[:])

                    if Bin1==Bin2:
                        # add noise to diag elems
                        Cl_GG[nl,Bin1,Bin2] += self.noise['GG']
                    else:
                        # use symmetry
                        Cl_GG[nl,Bin2,Bin1] = Cl_GG[nl,Bin1,Bin2]

    # compute the GL component
    if 'WL_GCph_XC' in self.probe:
        Cl_LG = np.zeros((len(self.l_XC), self.nbin, self.nbin), 'float64')
        Cl_LG_int = np.zeros((len(self.l_XC), self.nbin, self.nbin, self.nzmax), 'float64')
        Cl_GL = np.zeros((len(self.l_XC), self.nbin, self.nbin), 'float64')
        for nl in range(len(self.l_XC)):
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    # no symmetry of non-diag elems
                    Cl_LG_int[nl,Bin1,Bin2,:] = W_L[:, Bin1] * W_G[:, Bin2]  * pk_sqrt_bf[nl,:] / H_z[:] / self.r[:] / self.r[:]
                    Cl_LG[nl,Bin1,Bin2] = trapz(Cl_LG_int[nl,Bin1,Bin2,:], self.z[:])

                    # symmetry of LG and GL
                    Cl_GL[nl,Bin2,Bin1] = Cl_LG[nl,Bin1,Bin2]

                    if Bin1==Bin2:
                        Cl_LG[nl,Bin1,Bin2] += self.noise['LG']
                        Cl_GL[nl,Bin1,Bin2] += self.noise['GL']

    ########################
    # Spline Cl
    ########################
    # Find C(l) for every integer l

    # Spline the Cls along l
    if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
        spline_LL = np.empty((self.nbin, self.nbin),dtype=(list,3))
        for Bin1 in range(self.nbin):
            for Bin2 in range(self.nbin):
                spline_LL[Bin1,Bin2] = list(itp.splrep(
                    self.l_WL[:], Cl_LL[:,Bin1,Bin2]))

    if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
        spline_GG = np.empty((self.nbin, self.nbin), dtype=(list,3))
        for Bin1 in range(self.nbin):
            for Bin2 in range(self.nbin):
                spline_GG[Bin1,Bin2] = list(itp.splrep(
                    self.l_GC[:], Cl_GG[:,Bin1,Bin2]))

    if 'WL_GCph_XC' in self.probe:
        spline_LG = np.empty((self.nbin, self.nbin), dtype=(list,3))
        spline_GL = np.empty((self.nbin, self.nbin), dtype=(list,3))
        for Bin1 in range(self.nbin):
            for Bin2 in range(self.nbin):
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

    for Bin1 in range(self.nbin):
        for Bin2 in range(self.nbin):
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

    # np.save("/home/zr657363/Cl.npy", Cov_theory)
                
    #######################
    # Create fiducial file
    #######################

    if self.fid_values_exist is False:
        # Store the values now, and exit.
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file_path)
        if self.probe_WL and self.probe_GC:
            np.savez_compressed(fid_file_path, Cov_observ=Cov_theory, Cov_observ_high=Cov_theory_high)
        else:
            np.savez_compressed(fid_file_path, Cov_observ=Cov_theory)
        # with open(fid_file_path, 'w') as fid_file:
        #     fid_file.write('# Fiducial parameters')
        #     for key, value in data.mcmc_parameters.items():
        #         fid_file.write(
        #             ', %s = %.5g' % (key, value['current']*value['scale']))
        #     fid_file.write('\n')
        #     if 'WL' in self.probe or 'GCph' in self.probe:
        #         for Bin1 in range(self.nbin):
        #             for Bin2 in range(self.nbin):
        #                 for nl in range(len(Cov_theory[:,0,0])):
        #                     fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
        #     if 'WL_GCph_XC' in self.probe:
        #         for Bin1 in range(2*self.nbin):
        #             for Bin2 in range(2*self.nbin):
        #                 for nl in range(len(Cov_theory[:,0,0])):
        #                     fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
        #         for Bin1 in range(self.nbin):
        #             for Bin2 in range(self.nbin):
        #                 for nl in range(len(Cov_theory_high[:,0,0])):
        #                     fid_file.write("%.55g\n" % Cov_theory_high[nl, Bin1, Bin2])
        print('\n')
        warnings.warn(
            "Writing fiducial model in %s, for %s likelihood\n\n" % (
                fid_file_path, self.name))
        return

    ######################
    # Compute likelihood
    ######################
    # Define cov theory and observ on the whole integer range of ell values

    chi2 = 0.

    self.lmax_XC = self.lmax_GC

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
                for i in range(2*self.nbin):
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
                for i in range(self.nbin):
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
            for i in range(self.nbin):
                newCov = np.copy(Cov_theory[index, :, :])
                newCov[:, i] = self.Cov_observ[index, :, i]
                det_cross += np.linalg.det(newCov)/det_theory

            chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - self.nbin)

    elif 'GCph' in self.probe:
        for index, ell in enumerate(ells_GC):
            det_theory = np.linalg.det(Cov_theory[index,:,:])
            det_observ = np.linalg.det(self.Cov_observ[index,:,:])

            det_cross = 0.
            for i in range(self.nbin):
                newCov = np.copy(Cov_theory[index, :, :])
                newCov[:, i] = self.Cov_observ[index, :, i]
                det_cross += np.linalg.det(newCov)/det_theory

            chi2 += (2.*ell+1.)*self.fsky*(det_cross + np.log(det_theory/det_observ) - self.nbin)

    return chi2