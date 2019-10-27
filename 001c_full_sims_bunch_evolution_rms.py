import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

from scipy.signal import savgol_filter

import os,sys
BIN = os.path.expanduser("./tools/")
sys.path.append(BIN)

import myfilemanager as mfm
import propsort as ps
import mystyle as ms

from scipy.constants import c as ccc

# Chromaticity no damper
folders_compare = [
    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_%.1f'%qqpp).replace('-', 'minus') for qqpp in [0, 2.5]]

# Chromaticity with damper
qp_list = [0, 2.5, 5, 7.5, 10., 12.5, 15.]
labels = ['%.1f'%qqpp for qqpp in qp_list]
folders_compare = [
    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_%.1f'%qqpp).replace('-', 'minus') for qqpp in qp_list]
i_start_list = [0] * len(qp_list)
fname = 'wrongQp_effect_with_damper'


qp_list = [0, 2.5, 5]#, 7.5, 10., 12.5, 15.]
labels = ['%.1f'%qqpp for qqpp in qp_list]
folders_compare = [
    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_%.1f'%qqpp) for qqpp in qp_list]
i_start_list = [0] * len(qp_list)
fname = 'Qp_effect_with_damper'

## Damper ON/OFF (Qp = 0)
#labels = ['Feedback %s'%ff for ff in ['OFF', 'ON']]
#folders_compare = [
#    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_0.0',
#    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_0.0']
#i_start_list = [700, 700]
#fname = 'ADT_effect_Qp0'


## Damper ON/OFF (Qp = 2.5)
#labels = ['Feedback %s'%ff for ff in ['OFF', 'ON']]
#folders_compare = [
#    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_2.5',
#    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_2.5']
#i_start_list = [3000, 5000]
#fname = 'ADT_effect_Qp2.5'

# # Octupole scan
# labels = '-6 -3.0 -1.5 0.0 1.5 3.0 6'.split()
# folders_compare = [
#     ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_%s_Qp_xy_0.0'%oo).replace('-', 'minus') for oo in labels]
# i_start_list = [0,0,0,0,0,0,0]
# i_start_list = None

## Low/high intensity
#labels = ['2.3e11 p/b', '1.2e11 p/b']
#folders_compare = [
#        '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_scan_intensity_1.2_2.3e11_VRFandBunchLength_3_8MV/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_%.1fe11ppb_VRF_3MV'%vv for vv in [2.3, 1.2]]
#i_start_list = None
#fname = 'intensity_effect'

labels = ['test']
folders_compare = [
    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_scan_intensity_1.2_2.3e11_VRFandBunchLength_3_8MV/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_2.3e11ppb_VRF_5MV']
'../ArcQuad_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_5MV']
i_start_list = [750]
fname = None

plt.close('all')

fig1 = plt.figure(1, figsize=(8/1.3,6*1.5/1.3))
ax11 = fig1.add_subplot(3,1,1)
ax12 = fig1.add_subplot(3,1,2, sharex=ax11)
ax13 = fig1.add_subplot(3,1,3, sharex=ax11)

for ifol, folder in enumerate(folders_compare):

    print('Folder %d/%d'%(ifol, len(folders_compare)))

    folder_curr_sim = folder
    sim_curr_list = ps.sort_properly(glob.glob(folder_curr_sim+'/bunch_evolution_*.h5'))
    ob = mfm.monitorh5list_to_obj(sim_curr_list)

    sim_curr_list_slice_ev = ps.sort_properly(glob.glob(folder_curr_sim+'/slice_evolution_*.h5'))
    ob_slice = mfm.monitorh5list_to_obj(sim_curr_list_slice_ev, key='Slices', flag_transpose=True)

    w_slices = ob_slice.n_macroparticles_per_slice
    wx = ob_slice.mean_x * w_slices
    rms_x = np.sqrt(np.mean((ob_slice.mean_x * w_slices)**2, axis=0))
    mask_zero = ob.epsn_x > 0.

    ax11.plot(ob.mean_x[mask_zero]*1e3, label=labels[ifol])
    ax12.plot(ob.epsn_x[mask_zero]*1e6)
    ax13.plot(savgol_filter(rms_x[mask_zero], 21, 3))


    if i_start_list is not None:
        i_start = i_start_list[ifol]
        fig10 = plt.figure(10+ifol)
        for i_trace in range(i_start, i_start+15):
            wx_trace_filtered = savgol_filter(wx[:,i_trace], 51, 3)
            mask_filled = ob_slice.n_macroparticles_per_slice[:,i_trace]>0
            plt.plot(ob_slice.mean_z[mask_filled, i_trace], wx_trace_filtered[mask_filled])

for ax in [ax11, ax12, ax13]:
    ax.grid(True, linestyle='--', alpha=0.5)

ax13.set_xlabel('Turn')
ax13.set_ylabel('Intrabunch activity')
ax12.set_ylabel('Transverse emittance [um]')
ax11.set_ylabel('Transverse position [mm]')

leg = ax11.legend(prop={'size':10})
if fname is not None:
    fig1.savefig(fname+'.png', dpi=200)

import sys
sys.path.append('./NAFFlib')

<<<<<<< HEAD
figfft = plt.figure(300)
axfft = figfft.add_subplot(111)


=======
# I Try a global fft
figfft = plt.figure(300)
axfft = figfft.add_subplot(111)
>>>>>>> c4373d695896f8293a14819b5ab3eb7b87e59343
fftx = np.fft.rfft(ob.mean_x[mask_zero])
qax = np.fft.rfftfreq(len(ob.mean_x[mask_zero]))
axfft.semilogy(qax, np.abs(fftx))

<<<<<<< HEAD
import NAFFlib as nl

n_wind = 100
=======
# I try some NAFF on the centroid
import NAFFlib as nl

n_wind = 500
>>>>>>> c4373d695896f8293a14819b5ab3eb7b87e59343
N_lines = 10
freq_list = []
ampl_list = []

x_vect = ob.mean_x[mask_zero]
N_samples = len(x_vect)

for ii in range(N_samples):
    if ii < n_wind/2:
        continue
    if ii > N_samples-n_wind/2:
        continue

    freq, a1, a2 = nl.get_tunes(
            x_vect[ii-n_wind/2 : ii+n_wind/2], N_lines)
    freq_list.append(freq)
    ampl_list.append(np.abs(a1))

fignaff = plt.figure(301)
axnaff = fignaff.add_subplot(111)

mpbl = axnaff.scatter(x=np.array(N_lines*[np.arange(len(freq_list))]).T,
<<<<<<< HEAD
    y=np.array(freq_list), c=(np.array(ampl_list)), vmax=0.001*np.max(ampl_list),
    s=1)
plt.colorbar(mpbl)
=======
    y=np.array(freq_list), c=(np.array(ampl_list)), vmax=1*np.max(ampl_list),
    s=1)
plt.colorbar(mpbl)

L_zframe = np.max(ob_slice.mean_z[:, 0]) - np.min(ob_slice.mean_z[:, 0]) 
# I try some FFT on the slice motion
figffts = plt.figure(302)
axffts = figffts.add_subplot(111)
ffts = np.fft.fft(wx, axis=0) 
n_osc_axis = np.arange(ffts.shape[0])*4*ob.sigma_z[0]/L_zframe
axffts.pcolormesh(np.arange(wx.shape[1]), n_osc_axis, np.abs(ffts))
axffts.set_ylim(0, 5)

# I try a double fft
figfft2 = plt.figure(303)
axfft2 = figfft2.add_subplot(111)
fft2 = np.fft.fft(ffts, axis=1) 
q_axis_fft2 = np.arange(0, 1., 1./wx.shape[1]) 
axfft2.pcolormesh(q_axis_fft2,
        n_osc_axis, np.abs(fft2))
axfft2.set_ylabel('N. oscillations in 4 sigmaz')
axfft2.set_ylim(0, 5)
axfft2.set_xlim(0.25, .30)

# Plot time evolution of most unstable "mode"
i_mode = np.argmax(
        np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :-50]), axis=1)\
      - np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :50]), axis=1))
fig1mode = plt.figure(304)
ax1mode = fig1mode.add_subplot(111)
ax1mode.plot(np.real(ffts[i_mode, :]), label = 'cos comp.')
ax1mode.plot(np.imag(ffts[i_mode, :]), alpha=0.5, label='sin comp.')
ax1mode.legend(loc='best')
# These are the sin and cos components
# (r+ji)(cos + j sin) + (r-ji)(cos - j sin)=
# r cos + j r sin + ji cos - i sin | + r cos -j r sin -jicos -i sin = 
# 2r cos - 2 i sin

>>>>>>> c4373d695896f8293a14819b5ab3eb7b87e59343
plt.show()


