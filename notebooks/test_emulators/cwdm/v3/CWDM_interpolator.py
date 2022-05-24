import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import pickle
from emulator import gaussian_process
import os

HERE = os.path.dirname(os.path.realpath(__file__))

#======================
# USEFUL FUNCTIONS
#======================
def pca_reduce(X, nComp=2): 
    pca = PCA() 
    pca.fit(X)  
    mu = np.mean(X, axis=0) 
    coeff = pca.transform(X)[:,:nComp] 
    Comp = pca.components_[:nComp,:] 
    return {'coeff': coeff, 'pca': pca, 'mu': mu, 'Comp': Comp} 

def pca_expand(coeff, pca_dict=None, Comp=None, mu=0): 
    if coeff.ndim==1: coeff = coeff[None,:] 
    if pca_dict is not None: 
        mu = pca_dict['mu'] 
        Comp = pca_dict['pca'].components_ 
    nComp    = coeff.shape[1] 
    Xreconst = np.dot(coeff, Comp[:nComp,:]) + mu 
    return Xreconst
#======================

#======================
# EMULATOR
#======================
def CWDM_suppression(k,z,M_wdm,f_wdm,emulator_root=HERE+'/CWDM_emulator_files'):

    # Files
    filename_emulator_PCA = emulator_root+'/emulator_Pk_suppression_PCA_fwdm_Mwdm_z.pkl'
    filename_info_PCA     = emulator_root+'/PCAinfo_Pk_suppression_PCA_fwdm_Mwdm_z.pkl'
    k_emu_file            = emulator_root+'/k_values_list.npy'

    # Arrays
    k = np.atleast_1d(k)
    z = np.atleast_1d(z)
    assert isinstance(M_wdm,float), "M_wdm must be a float"
    assert isinstance(f_wdm,float), "f_wdm must be a float"
    if M_wdm>3.0: M_wdm=3.0
    if   f_wdm>1.:  raise ValueError("f_wdm must be smaller than or equal to 1")
    elif f_wdm==0.: return np.ones((len(z),len(k)))
    elif f_wdm<0.:  raise ValueError("f_wdm must be larger than 0")
    # Scales used
    k_emu = np.load(k_emu_file)
    # Load files (emulator and PCAs)
    ypca1_dict = pickle.load(open(filename_info_PCA,'rb'))
    # Load models
    gpr_load = gaussian_process.GPR_GPy()
    gpr_load.load_model(filename_emulator_PCA)
    # Parameters to compute (f_wdm, M_wdm, z)
    parameters = np.array([[f_wdm, M_wdm, z[i]] for i in range(len(z))])
    # Compute
    y_pred1, y_std1  = gpr_load.predict(parameters, return_std=True)
    supp_predicted   = pca_expand(y_pred1, pca_dict=ypca1_dict)
    # Manually set to 1 where suppression is above 1
    #supp_predicted[np.where(supp_predicted>1)] = 1.
    # Interpolation in (k,z)
    if len(z) > 1:
        kind_of_interp = 'cubic' if (len(z)>3) and (len(k))>3 else 'linear'
        supp_predicted_interp = si.interp2d(k_emu,z,supp_predicted,kind_of_interp,
                                            fill_value=0.,bounds_error=False)
        return supp_predicted_interp(k,z)
    else:
        kind_of_interp = 'cubic' if (len(k))>3 else 'linear'
        supp_predicted_interp = si.interp1d(k_emu,supp_predicted,kind_of_interp,
                                            fill_value='extrapolate',bounds_error=False)
        return np.atleast_2d(supp_predicted_interp(k))

#======================
