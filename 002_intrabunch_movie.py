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

N_prev = 20

# Damper ON
label = 'Qp2.5_ADT_ON'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_2.5'
i_start = 0
i_end = 6000


# label = 'Qp0_ADT_ON'
# folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_014/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_5e3_eMPs_500e3_damper_10turns_scan_chromaticity_minus2.5_20_intensity_1.2e11_2.3e11ppb/simulations_PyPARIS/Inj_ArcQuad_T0_x_slices_750_seg_8_MPslice_5e3_eMPs_250e3_length_7_VRF_4MV_damper_10turns_intensity_1.2e11ppb_Qp_xy_0.0'
# i_start = 0
# i_end = 1000

# Q'=12.5
label = 'Qp12.5_ADT_ON'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_12.5'
i_start = 0
i_end = 8000

# Q'= 0.
label = 'Qp0.0_ADT_ON'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_0.0'
i_start = 0
i_end = 430

# Q'= 5.
label = 'Qp5.0_ADT_ON'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_5.0'
i_start = 0
i_end = 2200

# Q'= 2.5
label = 'Qp2.5_ADT_ON'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus3_3_chromaticity_minus2.5_20/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_2.5'
i_start = 0
i_end = 670

# SEY 1.3 3MV ADT OFF
label = 'sey_1.3_3MV_ADT_OFF'
folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_013/injection_450GeV_arcQuad_1.2e11ppb_en_2.5um_1/convergence_studies_inj_arcQuad_scan_slices/simulations_PyPARIS/ArcQuad_T0_x_slices_500_segments_8_MPslice_5e3_eMPs_5e5_length_07_VRF_3MV'
i_start = 0
i_end = 500


plt.close('all')

folder_curr_sim = folder

sim_curr_list_slice_ev = ps.sort_properly(glob.glob(folder_curr_sim+'/slice_evolution_*.h5'))
ob_slice = mfm.monitorh5list_to_obj(sim_curr_list_slice_ev, key='Slices', flag_transpose=True)

w_slices = ob_slice.n_macroparticles_per_slice
wx = ob_slice.mean_x * w_slices
wx_filtered = savgol_filter(wx, 21, 3, axis=0)

ylim = np.max(np.abs(wx_filtered[:, i_start:i_end]))

fig = plt.figure(1)

for i_turn in xrange(i_start, i_end):
    fig.clf()
    ax = fig.add_subplot(111)
    for ii, i_trace in enumerate(range(i_turn-N_prev, i_turn)):
        if i_trace<0:
            continue
        wx_trace_filtered = wx_filtered[:, i_trace]
        mask_filled = ob_slice.n_macroparticles_per_slice[:,i_trace]>0
        ax.plot(ob_slice.mean_z[mask_filled, i_trace], wx_trace_filtered[mask_filled],
                    color='grey', alpha=(0.99*(float(ii)/float(N_prev))))
    mask_filled = ob_slice.n_macroparticles_per_slice[:, i_turn] > 0
    plt.plot(ob_slice.mean_z[mask_filled, i_turn], wx_filtered[mask_filled, i_turn], color='k', lw=2)
    ax.set_xlabel('z [m]')
    ax.set_ylabel('Pickup signal [a.u.]')
    ax.set_ylim(-ylim, ylim)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle('Turn %d'%i_turn)
    fig.savefig(label + 'frame_%05d.png'%i_turn, dpi=180)

command = ' '.join([
        'ffmpeg',
         '-i %s'%(label + 'frame_%05d.png'),
         '-c:v libx264 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,setpts=3*PTS"',
        '-profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac %s.mp4'%label])
os.system(command)

plt.show()
