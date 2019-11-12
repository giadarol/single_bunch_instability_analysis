import os,sys
sys.path.append("tools")
sys.path.append("PyHEADTAIL")

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

from scipy.signal import savgol_filter

from PyPARIS_sim_class import LHC_custom

import myfilemanager as mfm
import propsort as ps
import mystyle as ms

from scipy.constants import c as ccc

########################################### Old stuff
##### Chromaticity no damper
####folders_compare = [
####    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_%.1f'%qqpp).replace('-', 'minus') for qqpp in [0, 2.5]]

##### Chromaticity with damper
####qp_list = [0, 2.5, 5, 7.5, 10., 12.5, 15.]
####labels = ['%.1f'%qqpp for qqpp in qp_list]
####folders_compare = [
####    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_%.1f'%qqpp).replace('-', 'minus') for qqpp in qp_list]
####i_start_list = [0] * len(qp_list)
####fname = 'wrongQp_effect_with_damper'


####qp_list = [0, 2.5, 5]#, 7.5, 10., 12.5, 15.]
####labels = ['%.1f'%qqpp for qqpp in qp_list]
####folders_compare = [
####    ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_%.1f'%qqpp) for qqpp in qp_list]
####i_start_list = [0] * len(qp_list)
####fname = 'Qp_effect_with_damper'

##### Damper ON/OFF (Qp = 0)
####labels = ['Feedback %s'%ff for ff in ['OFF', 'ON']]
####folders_compare = [
####    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_0.0',
####    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_0.0']
####i_start_list = [700, 700]
####fname = 'ADT_effect_Qp0'


##### Damper ON/OFF (Qp = 2.5)
####labels = ['Feedback %s'%ff for ff in ['OFF', 'ON']]
####folders_compare = [
####    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_intensity_1.2e11ppb_Qp_xy_2.5',
####    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_2.5']
####i_start_list = [3000, 5000]
####fname = 'ADT_effect_Qp2.5'

###### Low/high intensity
#####labels = ['2.3e11 p/b', '1.2e11 p/b']
#####folders_compare = [
#####        '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_scan_intensity_1.2_2.3e11_VRFandBunchLength_3_8MV/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_%.1fe11ppb_VRF_3MV'%vv for vv in [2.3, 1.2]]
#####i_start_list = None
#####fname = 'intensity_effect'

#### labels = ['SEY 1.4 inj nokick']
#### folders_compare = [
####     '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_016/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_intensity_1.2e11ppb_VRF_3_8MV_no_initial_kick/simulations_PyPARIS/ArcQuad_no_initial_kick_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_7MV/']
#### fname = None
#### i_start_list = None
###
#### labels = ['FLATTOP']
#### folders_compare = [
####     '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_007/convergence_studies_arcQuad_Tb_slices/convergence_wrt_slices_250_750_1000_blocked_grid_only_x/simulations_PyPARIS/transverse_grid_Tblocked_betaxy_100m_length0.16_slices_750/']
#### fname = None
#### i_start_list = None
###
#### label = 'SEY 1.3 inj'
#### folders_compare = [
####  '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_013/injection_450GeV_arcQuad_1.2e11ppb_en_2.5um_1/convergence_studies_inj_arcQuad_scan_slices/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_5e3_eMPs_5e5_length_07_VRF_6MV']
#### fname = None
#### i_start_list = None
########################################### end old stuff

# # Volvage scan SEY = 1.3
# VRF_array = np.arange(3, 8.1, 1)
# labels = ['SEY 1.3 - %.1f MV'%vv for vv in VRF_array]
# folders_compare = [
#     '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_013/injection_450GeV_arcQuad_1.2e11ppb_en_2.5um_1/convergence_studies_inj_arcQuad_scan_slices/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_5e3_eMPs_5e5_length_07_VRF_%.0fMV'%vv for vv in VRF_array]
# fname = 'sey1.3_vscan'
# i_start_list = None

# Volvage scan SEY = 1.4
VRF_array = np.arange(3, 8.1, 1)
labels = ['SEY 1.4 - %.1f MV'%vv for vv in VRF_array]
folders_compare = [
    '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_scan_intensity_1.2_2.3e11_VRFandBunchLength_3_8MV/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_%.0fMV'%vv for vv in VRF_array]
fname = 'sey1.4_vscan'
i_start_list = None

# # Q' scan
# Qp_array = np.arange(0., 12.55, 2.5)[-2:]
# labels = ["Q'=%.1f"%qp for qp in Qp_array]
# folders_compare = [
#     '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_%.1f'%qp for qp in Qp_array]
# i_start_list = None
# fname = None

flag_naff = False

def extract_info_from_sim_param(spfname):
    with open(spfname, 'r') as fid:
        lines = fid.readlines()

    ddd = {}
    # Extract V_RF
    for ll in lines:
        if '=' in ll:
            nn = ll.split('=')[0].replace(' ','')
            try:
                ddd[nn] = eval(ll.split('=')[-1])
            except:
                ddd[nn] = 'Failed!'
    return ddd

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

    pars = extract_info_from_sim_param(folder+'/Simulation_parameters.py')

    w_slices = ob_slice.n_macroparticles_per_slice
    wx = ob_slice.mean_x * w_slices / np.mean(w_slices)
    rms_x = np.sqrt(np.mean((ob_slice.mean_x * w_slices)**2, axis=0))
    mask_zero = ob.epsn_x > 0.

    ax11.plot(ob.mean_x[mask_zero]*1e3, label=labels[ifol])
    ax12.plot(ob.epsn_x[mask_zero]*1e6)
    intrabunch_activity = savgol_filter(rms_x[mask_zero], 21, 3)
    ax13.plot(intrabunch_activity)

    import sys
    sys.path.append('./NAFFlib')

    figfft = plt.figure(300)
    axfft = figfft.add_subplot(111)

    figffts = plt.figure(3000 + ifol, figsize=(1.7*6.4, 1.8*4.8))
    plt.rcParams.update({'font.size': 12})

    axwidth = .38
    pos_col1 = 0.1
    pos_col2 = 0.57
    pos_row1 = 0.63
    height_row1 = 0.3
    pos_row2 = 0.37
    height_row2 = 0.18
    pos_row3 = 0.07
    height_row3 = 0.22

    axffts = figffts.add_axes((pos_col1, pos_row1, axwidth, height_row1))
    axfft2 = figffts.add_axes((pos_col2, pos_row1, axwidth, height_row1), sharey=axffts)
    axcentroid = figffts.add_axes((pos_col1, pos_row2, axwidth, height_row2),
            sharex=axffts)
    ax1mode = figffts.add_axes((pos_col2, pos_row2, axwidth, height_row2),
            sharex=axcentroid)
    axtraces = figffts.add_axes((pos_col1, pos_row3, axwidth, height_row3))
    axtext = figffts.add_axes((pos_col2, pos_row3, axwidth, height_row3))

    #axtraces = plt.subplot2grid(fig=figffts, shape=(3,4), loc=(2,1), colspan=2)

    figffts.subplots_adjust(
        top=0.925,
        bottom=0.07,
        left=0.11,
        right=0.95,
        hspace=0.3,
        wspace=0.28)

    fftx = np.fft.rfft(ob.mean_x[mask_zero])
    qax = np.fft.rfftfreq(len(ob.mean_x[mask_zero]))
    axfft.semilogy(qax, np.abs(fftx), label=labels[ifol])

    # I try some NAFF on the centroid
    import NAFFlib as nl
    if flag_naff:

        n_wind = 50
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
            y=np.array(freq_list), c=(np.array(ampl_list)),
            vmax=1*np.max(ampl_list),
            s=1)
        plt.colorbar(mpbl)

    # Details
    L_zframe = np.max(ob_slice.mean_z[:, 0]) - np.min(ob_slice.mean_z[:, 0])
    # I try some FFT on the slice motion
    ffts = np.fft.fft(wx, axis=0)
    n_osc_axis = np.arange(ffts.shape[0])*4*ob.sigma_z[0]/L_zframe
    axffts.pcolormesh(np.arange(wx.shape[1]), n_osc_axis, np.abs(ffts))
    axffts.set_ylim(0, 5)
    axffts.set_ylabel('N. oscillations\nin 4 sigmaz')
    axffts.set_xlabel('Turn')

    # I try a double fft
    fft2 = np.fft.fft(ffts, axis=1)
    q_axis_fft2 = np.arange(0, 1., 1./wx.shape[1])
    axfft2.pcolormesh(q_axis_fft2,
            n_osc_axis, np.abs(fft2))
    axfft2.set_ylabel('N. oscillations\nin 4 sigmaz')
    axfft2.set_ylim(0, 5)
    axfft2.set_xlim(0.25, .30)
    axfft2.set_xlabel('Tune')

    axcentroid.plot(ob.mean_x[mask_zero]*1000)
    axcentroid.set_xlabel('Turn')
    axcentroid.set_ylabel('Centroid position [mm]')
    axcentroid.grid(True, linestyle='--', alpha=0.5)
    axcentroid.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Plot time evolution of most unstable "mode"
    i_mode = np.argmax(
            np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :-50]), axis=1)\
          - np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :50]), axis=1))
    ax1mode.plot(np.real(ffts[i_mode, :][mask_zero]), label = 'cos comp.')
    ax1mode.plot(np.imag(ffts[i_mode, :][mask_zero]), alpha=0.5, label='sin comp.')
    ax1mode.legend(loc='best', prop={'size':12})
    ax1mode.set_xlabel('Turn')
    ax1mode.set_ylabel('Most unstable mode')
    ax1mode.grid(True, linestyle='--', alpha=0.5)
    ax1mode.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax1mode.set_xlim(0, np.sum(mask_zero))

    for ax in [axcentroid, ax1mode]:
        ax.set_ylim(np.array([-1, 1])*np.max(np.abs(np.array(ax.get_ylim()))))

    tune_centroid = nl.get_tune(ob.mean_x[mask_zero])
    tune_1mode_re = nl.get_tune(np.real(ffts[i_mode, :]))
    tune_1mode_im = nl.get_tune(np.imag(ffts[i_mode, :]))

    N_traces = 15
    max_intr = np.max(intrabunch_activity)
    i_start = np.where(intrabunch_activity<0.3*max_intr)[0][-1] - N_traces
    # i_start = np.sum(mask_zero) - 2*N_traces
    for i_trace in range(i_start, i_start+15):
        wx_trace_filtered = savgol_filter(wx[:,i_trace], 31, 3)
        mask_filled = ob_slice.n_macroparticles_per_slice[:,i_trace]>0
        axtraces.plot(ob_slice.mean_z[mask_filled, i_trace],
                    wx_trace_filtered[mask_filled])

    axtraces.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    axtraces.grid(True, linestyle='--', alpha=0.5)
    axtraces.set_xlabel("z [m]")
    axtraces.set_ylabel("P.U. signal")
    axtraces.text(0.02, 0.02, 'Turns:\n%d - %d'%(i_start,
                i_start+N_traces-1),
            transform=axtraces.transAxes, ha='left', va='bottom')

    plt.suptitle(labels[ifol])

    # Get Qx Qs
    machine = LHC_custom.LHC(
              n_segments=1,
              machine_configuration=pars['machine_configuration'],
              beta_x=pars['beta_x'], beta_y=pars['beta_y'],
              accQ_x=pars['Q_x'], accQ_y=pars['Q_y'],
              Qp_x=pars['Qp_x'], Qp_y=pars['Qp_y'],
              octupole_knob=pars['octupole_knob'],
              optics_dict=None,
              V_RF=pars['V_RF']
              )
    Qs = machine.longitudinal_map.Q_s
    Qx = machine.transverse_map.accQ_x
    frac_qx, _ = math.modf(Qx)

    axtext.text(0.5, 0.5,
            'Tune machine: %.4f'%frac_qx +\
            '\nSynchrotron tune: %.3fe-3 (V_RF: %.1f MV)'%(Qs*1e3, pars['V_RF']*1e-6) +\
        '\nTune centroid: %.4f (%.2fe-3)\n'%(tune_centroid, 1e3*tune_centroid-frac_qx*1e3)+\
        'Tune mode (cos): %.4f (%.2fe-3)\n'%(tune_1mode_re, 1e3*tune_1mode_re-1e3*frac_qx) +\
        'Tune mode (sin): %.4f (%.2fe-3)'%(tune_1mode_im, 1e3*tune_1mode_im-1e3*frac_qx),
        size=12, ha='center', va='center')
    axtext.axis('off')
    # These are the sin and cos components
    # (r+ji)(cos + j sin) + (r-ji)(cos - j sin)=
    # r cos + j r sin + ji cos - i sin | + r cos -j r sin -jicos -i sin = 
    # 2r cos - 2 i sin

    if fname is not None:
        figffts.savefig(fname+'_' + labels[ifol].replace(
            ' ', '_').replace('=', '').replace('-_', '')+'.png', dpi=200)

for ax in [ax11, ax12, ax13, axfft]:
    ax.grid(True, linestyle='--', alpha=0.5)

ax13.set_xlabel('Turn')
ax13.set_ylabel('Intrabunch\nactivity')
ax12.set_ylabel('Transverse\nemittance [um]')
ax11.set_ylabel('Transverse\nposition [mm]')
fig1.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.18,
        right=0.955,
        hspace=0.2,
        wspace=0.2)


leg = ax11.legend(prop={'size':10})
legfft = axfft.legend(prop={'size':10})
if fname is not None:
    fig1.savefig(fname+'.png', dpi=200)

plt.show()
