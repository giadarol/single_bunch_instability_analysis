import sys
sys.path.append('tools')
sys.path.append("PyHEADTAIL")

import math

import myfilemanager as mfm

from PyPARIS_sim_class import LHC_custom

folder = '/afs/cern.ch/project/spsecloud/Sim_PyPARIS_015/inj_arcQuad_T0_seg_8_slices_500_MPsSlice_2500_eMPs_5e5_sey_1.4_VRF_4MV_damper_10turns_scan_intensity_1.2_2.3e11_octupole_minus6_6_chromaticity_minus2.5_20_FP/simulations_PyPARIS/damper_10turns_length_7_VRF_4MV_intensity_1.2e11ppb_oct_1.5_Qp_xy_10.0_FP'


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
frac_qx, _ = math.modf(Qx)


filename_footprint = 'footprint.h5'
ob = mfm.object_with_arrays_and_scalar_from_h5(
        folder + '/' + filename_footprint)

betax = machine.transverse_map.beta_x[0]
betay = machine.transverse_map.beta_y[0]
Jy = (ob.y_init**2 + (ob.yp_init*betay)**2)/(2*betay)
Jx = (ob.x_init**2 + (ob.xp_init*betax)**2)/(2*betax)

