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

plt.close('all')

folder_curr_sim = folder

sim_curr_list_slice_ev = ps.sort_properly(glob.glob(folder_curr_sim+'/slice_evolution_*.h5'))
ob_slice = mfm.monitorh5list_to_obj(sim_curr_list_slice_ev, key='Slices', flag_transpose=True)

w_slices = ob_slice.n_macroparticles_per_slice
wx = ob_slice.mean_x * w_slices
wx_filtered = savgol_filter(wx, 21, 3, axis=0)

fig = plt.figure(1)

for i_turn in xrange(2000, 5000):
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
    ax.set_ylabel('Position [mm]')
    ax.set_ylim(-1.5,1.5)
    fig.suptitle('Turn %d'%i_turn)
    fig.savefig('frame_%05d.png'%i_turn, dpi=180)

plt.show()
