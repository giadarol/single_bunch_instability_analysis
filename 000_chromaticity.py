import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import e as qe
from scipy.constants import m_p
from scipy.constants import c as clight

from PyHEADTAIL.machines.synchrotron import Synchrotron
import PyPIC.PyPIC_Scatter_Gather as psg

chromaticity = 15.
p0c_eV_c = 450e9
p0 = p0c_eV_c*qe/clight
n_macroparticles = 100000
intensity = 1e11
epsn_x = 2.5e-6
epsn_y = 2.5e-6
sigma_z = 0.1

N_turns = 350

ring = Synchrotron(optics_mode = 'smooth',
        charge = qe, mass = m_p, 
        p0 = p0, circumference = 27e3, n_segments=1,
        alpha_x = 0., beta_x = 100., D_x = 0.,
        alpha_y = 0., beta_y = 100., D_y = 0.,
        accQ_x = 62.05, accQ_y = 64.29,
        longitudinal_mode = 'linear',
        alpha_mom_compaction = 3.225e-04,
        Q_s = 5e-3, Qp_x=chromaticity, use_cython=True)

bunch = ring.generate_6D_Gaussian_bunch(n_macroparticles,
                intensity, epsn_x, epsn_y, sigma_z)

pic = psg.PyPIC_Scatter_Gather(x_min=-3*sigma_z, x_max=3*sigma_z, 
    y_min=-3*bunch.sigma_dp(), y_max=3*bunch.sigma_dp(),
    dx=1e-2, dy=1e-4)

pic.scatter(x_mp=bunch.z, y_mp=bunch.dp,
        nel_mp=bunch.particlenumber_per_mp*np.ones_like(bunch.x), charge=1.)

plt.close('all')
fig1 = plt.figure(1)
ax11 = fig1.add_subplot(111)
ax11.pcolormesh(pic.xg, pic.yg, pic.rho.T)
fig1.suptitle('Longitudinal phase space density')

# Let's do some tracking
bunch.x += 1e-3

x_maps = []
for i_turns in range(N_turns):
    pic.scatter(x_mp=bunch.z, y_mp=bunch.dp,
        nel_mp=bunch.particlenumber_per_mp*bunch.x, charge=1.)
    x_maps.append(pic.rho.copy())
    
    ring.track(bunch)

x_maps = np.array(x_maps)
x_pu_signals = np.mean(x_maps, axis=2)
x_centroid = np.mean(x_pu_signals, axis=1)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(x_centroid)

N_traces = 50

fig3 = plt.figure(3, figsize=(6.4,1.5*4.8))
for i_turn in range(N_turns):
    fig3.clf()    
    ax31 = plt.subplot2grid(shape=(5, 1), loc=(0, 0))
    ax32 = plt.subplot2grid(shape=(5, 1), loc=(1, 0), rowspan=2)
    ax33 = plt.subplot2grid(shape=(5, 1), loc=(3, 0), rowspan=2)
    
    ax31.plot(x_centroid)
    ax31.axvline(x=i_turn)
    mpbl = ax33.pcolormesh(pic.xg, pic.yg, x_maps[i_turn, :, :].T,
            vmax=np.max(x_maps), vmin=np.min(x_maps))
    plt.colorbar(mpbl, orientation='horizontal')
    for ii in range(i_turn-N_traces, i_turn):
        if ii<0:
            continue
        ax32.plot(pic.xg, x_pu_signals[ii, :],
            color='grey', alpha=(0.99*(float(ii)/float(N_traces))))
    ax32.plot(pic.xg, x_pu_signals[i_turn, :], color='k',
            linewidth=2.)

    ax32.set_ylim(1.05*np.min(x_pu_signals), 1.05*np.max(x_pu_signals))
    
    fig3.subplots_adjust(bottom=.06, top=.9, hspace=.5)
    fig3.savefig('turn%04d.png'%i_turn, dpi=200)
plt.show()




