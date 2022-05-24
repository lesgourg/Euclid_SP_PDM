import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import pickle
import matplotlib
import matplotlib.cm as cm
from emulator import gaussian_process
plt.rc('text',usetex=True)
plt.rc('font',size=25,family='serif')
from CWDM_interpolator import CWDM_suppression


# Scales and redshifts
k_test = np.geomspace(1e-4, 30., 301)
z_test = np.linspace(0., 3.5, 36)

# Mass (in keV) and fraction
M_test = 0.06
f_test = 0.02

#======================
# TEST
#======================
# Generate
supp_th_PCA = CWDM_suppression(k=k_test,z=z_test,M_wdm=M_test,f_wdm=f_test,
                               emulator_root='CWDM_emulator_files')

def color_map_color(value, cmap_name='rainbow', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

plt.figure(figsize=(10,7))
L,B,R,T=0.15,0.15,0.95,0.90
plt.subplots_adjust(L,B,R,T,0,0)
plt.title('$f_\mathrm{wdm} = %.3f, M_\mathrm{wdm} = %.3f$ keV, $z = %.1f \\rightarrow %.1f \ (\Delta z = %.1f)$' %(f_test, M_test, z_test.min(), z_test.max(),np.diff(z_test)[0]),fontsize=20)
for i in range(len(z_test)):
    plt.semilogx(k_test, supp_th_PCA[i], color=color_map_color(i,vmax=len(z_test)))
plt.xlim(1e-2,10.)
plt.ylim(0.96, 1.01)
plt.xlabel('$k \ [h/\mathrm{Mpc}]$')
plt.ylabel('$P_\mathrm{CWDM}(k) / P_\mathrm{\Lambda CDM}(k)$')
plt.savefig('Pk_suppression_fwdm_%.3f_Mwdm_%.3f.png' %(f_test,M_test))
#plt.show()
#======================

