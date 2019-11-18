import sys
sys.path.append('tools')
sys.path.append("PyHEADTAIL")

import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm

import myfilemanager as mfm

from PyPARIS_sim_class import LHC_custom

fname_root = 'sey1.4_4MV_QP0_octscan'
# fname_root = None
octknob_vect = [-6, -3, -1.5, 0., 1.5, 3, 6]
folders = ['/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_%.1f_Qp_xy_0.0_FP'%oo for oo in octknob_vect]
for iff, ff in enumerate(folders): # some fixes
    folders[iff] = ff.replace('6.0', '6').replace('-', 'minus')
leg_labels = None
labels = ['Koct = %.1f'%oo for oo in octknob_vect]
cmap = plt.cm.rainbow

# fname_root = 'sey1.3_4MV_QP0_octscan'
# # fname_root = None
# octknob_vect = [-6, -3, -1.5, 0., 1.5, 3, 6]
# folders = ['/afs/cern.ch/project/spsecloud/Sim_PyPARIS_016/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.3_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_%.1f_Qp_xy_0.0_FP'%oo for oo in octknob_vect]
# for iff, ff in enumerate(folders): # some fixes
#     folders[iff] = ff.replace('6.0', '6').replace('-', 'minus')
# leg_labels = None
# labels = ['Koct = %.1f'%oo for oo in octknob_vect]
# cmap = plt.cm.rainbow

# fname_root = 'sey0.0_4MV_QP0_octscan'
# # fname_root = None
# octknob_vect = [-6, -3, -1.5, 1.5, 3, 6]
# folders = ['/afs/cern.ch/project/spsecloud/Sim_PyPARIS_016/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_VRF_4MV_damper_10turns_scan_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_oct_%.1f_Qp_xy_0.0_FP'%oo for oo in octknob_vect]
# for iff, ff in enumerate(folders): # some fixes
#     folders[iff] = ff.replace('6.0', '6').replace('-', 'minus')
# leg_labels = None
# labels = ['Koct = %.1f'%oo for oo in octknob_vect]
# cmap = plt.cm.rainbow

octup = -6.
fname_root = 'ecloud_effect_oct_%.1f'%octup
# fname_root = None
folders = ['/afs/cern.ch/project/spsecloud/Sim_PyPARIS_016/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_VRF_4MV_damper_10turns_scan_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_oct_%.1f_Qp_xy_0.0_FP'%octup,
'/afs/cern.ch/project/spsecloud/Sim_PyPARIS_016/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.3_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_%.1f_Qp_xy_0.0_FP'%octup,
    ]
for iff, ff in enumerate(folders): # some fixes
    folders[iff] = ff.replace('6.0', '6').replace('-', 'minus')
leg_labels = ['No e-cloud', 'SEY 1.3']
labels = leg_labels
cmap = lambda vv: plt.cm.tab10(int(vv*len(folders)))

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

plt.close('all')
plt.rcParams.update({'font.size': 12})

figglob = plt.figure(1)
axglob = figglob.add_subplot(111)
axdistrlist = []
figfplist = []
for ifol, folder in enumerate(folders):
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

    Qx_min = frac_qx -  0.03
    Qy_min = frac_qy -  0.03
    Qx_max_cut = frac_qx + 0.05
    Qy_max_cut = frac_qy + 0.05

    fig1 = plt.figure(1000+ifol, figsize=(6.4*1.1, 4.8*1.4))
    figfplist.append(fig1)

    ax1 = fig1.add_subplot(111)
    mpbl1 = ax1.scatter(np.abs(ob.qx_i), np.abs(ob.qy_i),
            c =ob.z_init*1e2, marker='.', edgecolors='none', vmin=-32, vmax=32)
    ax1.plot([frac_qx], [frac_qy], '*k', markersize=10)
    ax1.set_xlabel('Q$_x$')
    ax1.set_ylabel('Q$_y$')
    ax1.set_aspect(aspect='equal', adjustable='datalim')
    ax1.set(xlim=(Qx_min, Qx_max_cut), ylim=(Qy_min, Qy_max_cut))
    ax1.grid(True, linestyle='--', alpha=0.5)

    divider = make_axes_locatable(ax1)
    axhistx = divider.append_axes("top", size=1.2, pad=0.25, sharex=ax1)
    axcb = divider.append_axes("right", size=0.3, pad=0.1)
    axhistx.grid(True, linestyle='--', alpha=0.5)
    obstat = sm.nonparametric.KDEUnivariate(ob.qx_i)
    obstat.fit(bw=10e-4)
    q_axis = np.linspace(Qx_min, Qx_max_cut, 1000)
    axhistx.plot(q_axis, obstat.evaluate(q_axis))
    axhistx.fill_between(x=q_axis, y1=0, y2=obstat.evaluate(q_axis), alpha=0.5)
    if leg_labels is None:
        lll ='%.1f'%machine.i_octupole_focusing
    else:
        lll = leg_labels[ifol]
    axglob.plot(q_axis, obstat.evaluate(q_axis),
            label=lll,
            linewidth=2.,
            color=cmap(float(ifol)/float(len(folders))))
    axdistrlist.append(axhistx)
    plt.colorbar(mpbl1, cax=axcb)

    fig2 = plt.figure(2000+ifol)
    ax2 = fig2.add_subplot(111)
    mpbl = ax2.scatter(ob.z_init*1e2,
            np.abs(ob.qx_i)-frac_qx, c =Jx,
            marker='.', edgecolors='none', vmin=0, vmax=8e-9)
    cb = plt.colorbar(mpbl)
    cb.ax.set_ylabel('Transverse action')
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-0.0, 3e-2)
    ax2.set_xlabel('z [cm]')
    ax2.set_ylabel('$\Delta$Qx', labelpad=5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.155,
            right=0.965,
            hspace=0.2,
            wspace=0.2)
    # sigma_x = np.sqrt(pars['epsn_x']*betax/machine.betagamma)
    # sigma_y = np.sqrt(pars['epsn_y']*betay/machine.betagamma)
    # mask_small_amplitude = np.sqrt(
    #         (ob.x_init/sigma_x)**2 +(ob.x_init/sigma_x)**2) < 0.2
    # z_small = ob.z_init[mask_small_amplitude]
    # qx_small = ob.qx_i[mask_small_amplitude]
    # ax2.plot(z_small*1e2, qx_small - frac_qx, 'k.', markersize=10)

    for ff in [fig1, fig2]:
        ff.suptitle(labels[ifol] + ' - I$_{LOF}$=%.1fA'%machine.i_octupole_focusing)

if leg_labels is None:
    legtitle = 'I$_{LOF}$'
else:
    legtitle = None
axglob.legend(loc='best', title=legtitle)
axglob.grid(True, linestyle='--', alpha=0.5)
axglob.set_ylim(bottom=0)
axglob.set_ylabel('Density [a.u.]')
axglob.set_xlabel('Q$_x$')

for aa in axdistrlist:
    aa.set_ylim(axglob.get_ylim())
    aa.set_ylabel('Density [a.u.]')

if fname_root is not None:
    figglob.savefig(fname_root+'_spreads.png', dpi=200)
    for ff, ll in zip(figfplist, labels):
        ff.savefig(fname_root+'_'+ll.replace(' ', '_').replace('_=', '')+'.png', dpi=200)

plt.show()
