#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <spline.h>
// splining algorithm taken from https://kluge.in-chemnitz.de/opensource/spline/
#include <Eigen/Dense>
// for more information visit https://eigen.tuxfamily.org/

using namespace std;
using namespace Eigen;



double trapz (const double *Cl, const double *z, const int n) {
    double result = 0;
    for (int i = 0; i < n-1; i++)
        result += (Cl[i] + Cl[i+1])/2. * (z[i+1] - z[i]);
    return result;
}



extern "C" double chi2_ac (
    int nzmax,
    int nbin,
    int lbin,
    int lmin,
    int lmax,
    double noise,
    double fsky,
    const double *l_ac,
    const double *W,
    const double *pk,
    const double *H_z,
    const double *r,
    const double *z,
    const double *Cov_observ_py,
    bool scale_dependent_f
) {
    vector<vector<vector<double>>> Cl (nbin, vector<vector<double>> (nbin, vector<double> (lbin)));

    // Loop over all lbins, Bin1 and Bin2
    // Calculate Cl at those points
    double* Cl_int = new double[nzmax];
    // in case of a scale-dependent growth factor:
    int scale_dependence_array_offset = 0;
    for (int i = 0; i < lbin; i++) {
        if (scale_dependent_f) scale_dependence_array_offset = i*nzmax*nbin;
        for (int j = 0; j < nbin; j++) {
            for (int k = j; k < nbin; k++) {
                for (int nz = 0; nz < nzmax; nz++) {
                    Cl_int[nz] = 1.* W[scale_dependence_array_offset + nz*nbin + j] * W[scale_dependence_array_offset + nz*nbin + k] * pk[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                }
                double integrated_Cl = trapz(Cl_int, z, nzmax);
                Cl[j][k][i] = integrated_Cl;
            }
            Cl[j][j][i] += noise;
        }
    }
    delete[] Cl_int;
    
    // Spline Cls onto all integer ls between lmin and lmax
    int num_ls = lmax - lmin + 1;
    vector<double> ells (lbin, 0);
    for (int i = 0; i < lbin; i++) ells[i] = l_ac[i];
    vector<vector<vector<double>>> Cov_theory_vec (num_ls, vector<vector<double>> (nbin, vector<double> (nbin, 0)));
    #pragma omp parallel firstprivate(num_ls, lmin, nbin)
    {
        #pragma omp for
        for (int j = 0; j < nbin; j++)
            for (int k = j; k < nbin; k++) {
                tk::spline s(ells, Cl[j][k]);
                for (int i = 0; i < num_ls; i++) {
                    double splined = s(lmin + i);
                    Cov_theory_vec[i][j][k] = splined;
                    Cov_theory_vec[i][k][j] = splined;
                }
            }
    }

    // Calculate chi2
    // LU decomposition of the matrices is used for the calculation of the determinants.
    double chi2 = 0;
    #pragma omp parallel reduction(+:chi2) firstprivate(num_ls, lmin, nbin, fsky)
    {
        #pragma omp for
        for (int i = 0; i < num_ls; i++) {
            Eigen::MatrixXd Cov_observ (nbin, nbin);
            Eigen::MatrixXd Cov_theory (nbin, nbin);
            for (int j = 0; j < nbin; j++) for (int k = 0; k < nbin; k++) {
                Cov_theory (j,k) = Cov_theory_vec [i][j][k];
                Cov_observ (j,k) = Cov_observ_py [i*nbin*nbin + j*nbin + k];
            }
            PartialPivLU<MatrixXd> Cov_theory_LU = PartialPivLU<MatrixXd>(Cov_theory);
            PartialPivLU<MatrixXd> Cov_observ_LU = PartialPivLU<MatrixXd>(Cov_observ);
            double det_theory = Cov_theory.determinant();
            double det_observ = Cov_observ.determinant();
            double det_cross = 0;
            for (int j = 0; j < nbin; j++) {
                MatrixXd Cov_theory_cross = Cov_theory;
                for (int k = 0; k < nbin; k++)
                    Cov_theory_cross (k, j) = Cov_observ (k, j);
                PartialPivLU<MatrixXd> Cov_theory_cross_LU = PartialPivLU<MatrixXd>(Cov_theory_cross);
                det_cross += 1.*Cov_theory_cross_LU.determinant()/det_theory;
            }
            chi2 += (2.*(lmin+i)+1.) * fsky * (det_cross + log(det_theory/det_observ) - nbin);
        }
    }

    return chi2;
}



extern "C" double chi2_xc (
    const int nzmax,
    const int nbin,
    const int lbin_high,
    const int lbin_low,
    const int lmin,
    const int lmax_high,
    const int lmax_low,
    const double noise_high,
    const double noise_low,
    const double fsky,
    const double *l_high,
    const double *l_low,
    const double *W_L,
    const double *W_G,
    const double *pk,
    const double *pk_no_bf,
    const double *pk_sqrt_bf,
    const double *H_z,
    const double *r,
    const double *z,
    const double *Cov_observ_py,
    const double *Cov_observ_high_py,
    bool scale_dependent_f_high,
    bool scale_dependent_f_low
) {
    vector<vector<vector<double>>> Cl_LL (nbin, vector<vector<double>> (nbin, vector<double> (lbin_high)));
    vector<vector<vector<double>>> Cl_LG (nbin, vector<vector<double>> (nbin, vector<double> (lbin_low)));
    vector<vector<vector<double>>> Cl_GG (nbin, vector<vector<double>> (nbin, vector<double> (lbin_low)));
    // here L and G simply stand for "the probe with the higher lmax" and "the probe with the lower lmax"

    // loop over all lbins, Bin1 and Bin2 and calculate Cl at those points
    // don't worry, the compiler fixes the if statements inside the for loops :)
    double* Cl_int = new double[nzmax];
    // auto-correlation of probe with higher lmax
    for (int i = 0; i < lbin_high; i++) {
        for (int j = 0; j < nbin; j++) {
            for (int k = j; k < nbin; k++) {
                for (int nz = 0; nz < nzmax; nz++) {
                    if (scale_dependent_f_high)
                        Cl_int[nz] = 1.* W_L[i*nzmax*nbin + nz*nbin + j] * W_L[i*nzmax*nbin + nz*nbin + k] * pk[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                    else
                        Cl_int[nz] = 1.* W_L[nz*nbin + j] * W_L[nz*nbin + k] * pk[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                }
                double integrated_Cl = 1.* trapz(Cl_int, z, nzmax);
                Cl_LL[j][k][i] = integrated_Cl;
            }
            Cl_LL[j][j][i] += noise_high;
        }
    }
    // auto-correlation of probe with lower lmax
    for (int i = 0; i < lbin_low; i++) {
        for (int j = 0; j < nbin; j++) {
            for (int k = j; k < nbin; k++) {
                for (int nz = 0; nz < nzmax; nz++) {
                    if (scale_dependent_f_low)
                        Cl_int[nz] = 1.* W_G[i*nzmax*nbin + nz*nbin + j] * W_G[i*nzmax*nbin + nz*nbin + k] * pk_no_bf[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                    else
                        Cl_int[nz] = 1.* W_G[nz*nbin + j] * W_G[nz*nbin + k] * pk_no_bf[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                }
                double integrated_Cl = 1.* trapz(Cl_int, z, nzmax);
                Cl_GG[j][k][i] = integrated_Cl;
            }
            Cl_GG[j][j][i] += noise_low;
        }
    }
    // cross-correlation. Note that the resulting matrices aren't symmetrical.
    for (int i = 0; i < lbin_low; i++) {
        for (int j = 0; j < nbin; j++) {
            for (int k = 0; k < nbin; k++) {
                for (int nz = 0; nz < nzmax; nz++) {
                    if (scale_dependent_f_high)
                        Cl_int[nz] = 1.* W_L[i*nzmax*nbin + nz*nbin + j] * W_G[nz*nbin + k] * pk_sqrt_bf[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                    else if (scale_dependent_f_low)
                        Cl_int[nz] = 1.* W_L[nz*nbin + j] * W_G[i*nzmax*nbin + nz*nbin + k] * pk_sqrt_bf[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                    else
                        Cl_int[nz] = 1.* W_L[nz*nbin + j] * W_G[nz*nbin + k] * pk_sqrt_bf[i*nzmax + nz] / H_z[nz] / r[nz] / r[nz];
                }
                double integrated_Cl = 1.* trapz(Cl_int, z, nzmax);
                Cl_LG[j][k][i] = integrated_Cl;
            }
        }
    }
    delete[] Cl_int;

    // Spline Cls onto all integer ls between lmin and lmax
    int num_ls = lmax_low - lmin + 1;
    vector<double> ells_high (lbin_high, 0);
    vector<double> ells_low  (lbin_low, 0);
    for (int i = 0; i < lbin_high; i++) ells_high[i] = l_high[i];
    for (int i = 0; i < lbin_low;  i++) ells_low [i] = l_low[i];
    vector<vector<vector<double>>> Cov_theory_vec (num_ls,
        vector<vector<double>> (2*nbin, vector<double> (2*nbin, 0)));
    vector<vector<vector<double>>> Cov_theory_high_vec (lmax_high - num_ls,
        vector<vector<double>> (nbin, vector<double> (nbin, 0)));
    // ^ contains the covariance matrices for the probe with the higher lmax
    
    #pragma omp parallel firstprivate(lmax_high, lmax_low, lmin, nbin)
    {
        #pragma omp for
        for (int j = 0; j < nbin; j++) {
            for (int k = j; k < nbin; k++) {
                tk::spline LL(ells_high, Cl_LL[j][k]);
                tk::spline GG(ells_low,  Cl_GG[j][k]);
                for (int i = 0; i < num_ls; i++) {
                    double splined_LL = LL(lmin + i);
                    double splined_GG = GG(lmin + i);
                    Cov_theory_vec[i][j][k] = splined_LL;
                    Cov_theory_vec[i][k][j] = splined_LL;
                    Cov_theory_vec[i][j+nbin][k+nbin] = splined_GG;
                    Cov_theory_vec[i][k+nbin][j+nbin] = splined_GG;
                }
                for (int i = 0; i < lmax_high - num_ls; i++) {
                    double splined_LL = LL(num_ls + lmin + i);
                    Cov_theory_high_vec[i][j][k] = splined_LL;
                    Cov_theory_high_vec[i][k][j] = splined_LL;
                }
            }
            for (int k = 0; k < nbin; k++) {
                tk::spline LG(ells_low, Cl_LG[j][k]);
                for (int i = 0; i < num_ls; i++) {
                    double splined_LG = LG(lmin + i);
                    Cov_theory_vec[i][j][k+nbin] = splined_LG;
                    Cov_theory_vec[i][k+nbin][j] = splined_LG;
                }
            }
        }
    }

    // Compute likelihood
    // LU decomposition of the matrices is used for the calculation of the determinants.
    double chi2 = 0;
    #pragma omp parallel reduction(+:chi2) firstprivate(lmax_high, lmax_low, lmin, nbin, fsky)
    {
        #pragma omp for nowait
        for (int i = 0; i <= lmax_low-lmin; i++) {
            MatrixXd Cov_observ (2*nbin, 2*nbin);
            MatrixXd Cov_theory (2*nbin, 2*nbin);
            for (int j = 0; j < 2*nbin; j++) for (int k = 0; k < 2*nbin; k++) {
                Cov_theory (j,k) = Cov_theory_vec [i][j][k];
                Cov_observ (j,k) = Cov_observ_py [i*4*nbin*nbin + j*2*nbin + k];
            }
            PartialPivLU<MatrixXd> Cov_theory_LU = PartialPivLU<MatrixXd>(Cov_theory);
            PartialPivLU<MatrixXd> Cov_observ_LU = PartialPivLU<MatrixXd>(Cov_observ);
            double det_theory = Cov_theory.determinant();
            double det_observ = Cov_observ.determinant();
            double det_cross = 0;
            for (int j = 0; j < 2*nbin; j++) {
                MatrixXd Cov_theory_cross = Cov_theory;
                for (int k = 0; k < 2*nbin; k++)
                    Cov_theory_cross (k, j) = Cov_observ (k, j);
                PartialPivLU<MatrixXd> Cov_theory_cross_LU = PartialPivLU<MatrixXd>(Cov_theory_cross);
                det_cross += 1.*Cov_theory_cross_LU.determinant()/det_theory;
            }
            chi2 += (2.*(lmin+i)+1.) * fsky * (det_cross + log(det_theory/det_observ) - 2*nbin);
        }

        #pragma omp for
        for (int i = 0; i < lmax_high - lmax_low; i++) {
            MatrixXd Cov_observ_high (nbin, nbin);
            MatrixXd Cov_theory_high (nbin, nbin);
            for (int j = 0; j < nbin; j++) for (int k = 0; k < nbin; k++) {
                Cov_theory_high (j,k) = Cov_theory_high_vec [i][j][k];
                Cov_observ_high (j,k) = Cov_observ_high_py [i*nbin*nbin + j*nbin + k];
            }
            PartialPivLU<MatrixXd> Cov_theory_high_LU = PartialPivLU<MatrixXd>(Cov_theory_high);
            PartialPivLU<MatrixXd> Cov_observ_high_LU = PartialPivLU<MatrixXd>(Cov_observ_high);
            double det_theory = Cov_theory_high.determinant();
            double det_observ = Cov_observ_high.determinant();
            double det_cross = 0;
            for (int j = 0; j < nbin; j++) {
                MatrixXd Cov_theory_cross = Cov_theory_high;
                for (int k = 0; k < nbin; k++)
                    Cov_theory_cross (k, j) = Cov_observ_high (k, j);
                PartialPivLU<MatrixXd> Cov_theory_cross_LU = PartialPivLU<MatrixXd>(Cov_theory_cross);
                det_cross += 1.*Cov_theory_cross_LU.determinant()/det_theory;
            }
            chi2 += (2.*(lmax_low+i+1)+1.) * fsky * (det_cross + log(det_theory/det_observ) - nbin);
        }
    }

    return chi2;
}



extern "C" void W_gamma_int (
    const int nbin,
    const int nzmax,
    const double *eta_z,
    const double *r,
    const double *z,
    double *W_gamma
) {
    // Performs the integration in the calculation of W_L
    for (int bin = 0; bin < nbin; bin++)
        for (int nz = 0; nz < nzmax - 1; nz++) {
            double* integrand = new double[nzmax-nz];
            double* z_int     = new double[nzmax-nz];
            for (int i = nz; i < nzmax; i++) {
                integrand[i-nz] = 1.* eta_z[i*nbin + bin] * (r[i] - r[nz]) / r[i];
                z_int    [i-nz] = z[i];
            }
            W_gamma[nz*nbin + bin] = trapz (integrand, z_int, nzmax-nz);
            delete[] integrand;
            delete[] z_int;
        }
}
