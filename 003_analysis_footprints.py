import sys
sys.path.append('tools')
sys.path.append("PyHEADTAIL")

import math

import numpy as np
import matplotlib.pyplot as plt

import myfilemanager as mfm

from PyPARIS_sim_class import LHC_custom

folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_0.0_Qp_xy_0.0_FP'


def extract_info_from_sim_param(fname):
    with open(fname, 'r') as fid:
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

pars = extract_info_from_sim_param(folder+'/Simulation_parameters.py')
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
Qy = machine.transverse_map.accQ_y
frac_qx, _ = math.modf(Qx)
frac_qy, _ = math.modf(Qy)


filename_footprint = 'footprint.h5'
ob = mfm.object_with_arrays_and_scalar_from_h5(
        folder + '/' + filename_footprint)

betax = machine.transverse_map.beta_x[0]
betay = machine.transverse_map.beta_y[0]
Jy = (ob.y_init**2 + (ob.yp_init*betay)**2)/(2*betay)
Jx = (ob.x_init**2 + (ob.xp_init*betax)**2)/(2*betax)

Qx_min = frac_qx - frac_qx * 0.03
Qy_min = frac_qy - frac_qy * 0.03
Qx_max_cut = frac_qx + frac_qx * 0.1
Qy_max_cut = frac_qy + frac_qx * 0.1

plt.close('all')

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
s1 = ax1.scatter(np.abs(ob.qx_i), np.abs(ob.qy_i),
        c =ob.z_init*1e2, marker='.', edgecolors='none', vmin=-32, vmax=32)
ax1.plot([frac_qx], [frac_qy], '*k', markersize=10)
ax1.set_xlabel('Q$_x$')
ax1.set_ylabel('Q$_y$')
ax1.set_xlim([Qx_min, Qx_max_cut])
ax1.set_ylim([Qy_min, Qy_max_cut])
ax1.set_aspect(aspect=1, adjustable='box')


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
mpbl = ax2.scatter(ob.z_init*1e2,
        np.abs(ob.qx_i)-frac_qx, c =Jx,
        marker='.', edgecolors='none', vmin=0, vmax=8e-9)
plt.colorbar(mpbl)
ax2.set_xlim(-30, 30)
ax2.set_ylim(-0.0, 3e-2)
ax2.set_xlabel('z [cm]')
ax2.set_ylabel('$\Delta$Qx', labelpad=5)
ax2.grid(True, linestyle='--', alpha=0.5)

# sigma_x = np.sqrt(pars['epsn_x']*betax/machine.betagamma)
# sigma_y = np.sqrt(pars['epsn_y']*betay/machine.betagamma)
# mask_small_amplitude = np.sqrt(
#         (ob.x_init/sigma_x)**2 +(ob.x_init/sigma_x)**2) < 0.2
# z_small = ob.z_init[mask_small_amplitude]
# qx_small = ob.qx_i[mask_small_amplitude]
# ax2.plot(z_small*1e2, qx_small - frac_qx, 'k.', markersize=10)

plt.show()
