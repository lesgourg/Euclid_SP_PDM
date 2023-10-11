from ctypes import *
from time import time
import sys, os
import numpy as np
import numpy.ctypeslib as npc
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
try:
    _photo = CDLL(os.path.dirname(os.path.realpath(__file__)) + '/c++/euclid_photo.so')
except:
    print("Compiling shared library montepython/likelihoods/euclid_photometric_fast/c++/euclid_photo.so...")
    cpp_file_path = os.path.dirname(os.path.realpath(__file__)) + '/c++'
    os.system("g++ -shared -o " + cpp_file_path + "/euclid_photo.so " + cpp_file_path + "/euclid_photo.cpp -I " + cpp_file_path + " -fPIC -O3 -lgomp -fopenmp")
    try:
        _photo = CDLL(os.path.dirname(os.path.realpath(__file__)) + '/c++/euclid_photo.so')
    except:
        raise Exception("Compilation failed, please compile manually using ./montepython/likelihoods/euclid_photometric_fast/c++/compile.sh!")



##########################################
# WL and GCph auto-correlation functions #
##########################################



_photo.chi2_ac.restype = c_double
_photo.chi2_ac.argtypes = [
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_double,
    c_double,
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    c_bool
]

def chi2_WL (self, W_L, pk):
    if self.print_individual_times:
        self.chunk_time = time()
    chi2 = _photo.chi2_ac (
        self.nzmax,
        self.nbin,
        self.lbin,
        self.lmin,
        self.lmax_WL,
        self.WL_noise,
        self.fsky,
        self.l_high,
        W_L,
        pk,
        self.dzdr,
        self.r,
        self.z,
        self.Cov_observ,
        self.scale_dependent_f
    )
    if self.print_individual_times:
        print("calculate chi2 (C++):", time() - self.chunk_time, "s")
    return chi2

def chi2_GC (self, W_G, pk):
    if self.print_individual_times:
        self.chunk_time = time()
    chi2 = _photo.chi2_ac (
        self.nzmax,
        self.nbin,
        self.lbin,
        self.lmin,
        self.lmax_GC,
        self.GC_noise,
        self.fsky,
        self.l_high,
        W_G,
        pk,
        self.dzdr,
        self.r,
        self.z,
        self.Cov_observ,
        self.scale_dependent_f
    )
    if self.print_individual_times:
        print("calculate chi2 (C++):", time() - self.chunk_time, "s")
    return chi2



##################################
# 3x2 cross-correlation function #
##################################



_photo.chi2_xc.restype = c_double
_photo.chi2_xc.argtypes = [
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_double,
    c_double,
    c_double,
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    c_bool
]

def chi2_xc (self, W_L, W_G, pk):
    if self.print_individual_times:
        self.chunk_time = time()
    chi2 = _photo.chi2_xc (
        self.nzmax,
        self.nbin,
        len(self.l_high),
        len(self.l_low),
        self.lmin,
        self.lmax_WL,
        self.lmax_GC,
        self.WL_noise if self.lmax_WL > self.lmax_GC else self.GC_noise,
        # ^ WL noise
        self.GC_noise if self.lmax_WL > self.lmax_GC else self.WL_noise,
        # ^ GCph noise
        self.fsky,
        self.l_high,
        self.l_low,
        W_L,
        W_G,
        pk,
        self.dzdr,
        self.r,
        self.z,
        self.Cov_observ,
        self.Cov_observ_high,
        True if self.scale_dependent_f and (self.lmax_WL > self.lmax_GC) else False,
        True if self.scale_dependent_f and (self.lmax_GC > self.lmax_WL) else False
    )
    if self.print_individual_times:
        print("calculate chi2 (C++):", time() - self.chunk_time, "s")
    return chi2



##################################
# WL window function integration #
##################################



_photo.W_gamma_int.restype = c_void_p
_photo.W_gamma_int.argtypes = [
    c_int,
    c_int,
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64),
    npc.ndpointer(dtype=np.float64)
]

def W_gamma (self, cosmo):
    W_gamma = np.zeros((self.nzmax, self.nbin), 'float64')
    H0 = cosmo.h()/2997.92458
    # ^ (H_0 /c) in 1/Mpc
    _photo.W_gamma_int (
        self.nbin,
        self.nzmax,
        self.eta_z,
        self.r,
        self.z,
        W_gamma
    )
    W_gamma *= 3./2.*H0**2. *cosmo.Omega_m()*self.r[:,None]*(1.+self.z[:,None])
    return W_gamma