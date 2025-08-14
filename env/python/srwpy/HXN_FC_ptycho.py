#!/usr/bin/env python

import os
try:
    __IPYTHON__
    import sys
    del sys.argv[1:]
except:
    pass

try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from srwl_bl import *
    from srwl_uti_smp import *
    from uti_io_genesis import *
    from srwl_uti_detector import *
    #from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.srwl_uti_smp import *
    from srwpy.uti_plot import *
    from srwpy.srwl_uti_smp import *
    from srwpy.uti_io_genesis import *
    from srwpy.srwl_uti_detector import *

from time import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shutil
import copy

# import srwpy.srwl_bl
# import srwpy.srwlib
# import srwpy.srwlpy
# import math
# import srwpy.srwl_uti_smp

#------------------------------------------------------------------------------
def set_optics(v, names=None, want_final_propagation=True):
    el = []
    pp = []
    if not names:
        names = ['S1', 'S1_HCM', 'HCM', 'HCM_Before_DCM', 'DCM_C1', 'DCM_C2', 'After_DCM_HFM', 'HFM', 'HFM_VFM', 'VFM', 'VFM_VPM', 'VPM', 'After_VPM_Before_SSA', 'SSA', 'After_SSA_Before_VKB', 'VKB', 'After_VKB_Before_HKB', 'HKB', 'After_HKB_Focus', 'Sample', 'Watchpoint_Detector']
    for el_name in names:
        if el_name == 'S1':
            # S1: aperture 26.62m
            el.append(SRWLOptA(
                _shape=v.op_S1_shape,
                _ap_or_ob='a',
                _Dx=v.op_S1_Dx,
                _Dy=v.op_S1_Dy,
                _x=v.op_S1_x,
                _y=v.op_S1_y,
            ))
            pp.append(v.op_S1_pp)
        elif el_name == 'S1_HCM':
            # S1_HCM: drift 26.62m
            el.append(SRWLOptD(
                _L=v.op_S1_HCM_L,
            ))
            pp.append(v.op_S1_HCM_pp)
        elif el_name == 'HCM':
            # HCM: sphericalMirror 28.35m
            el.append(SRWLOptMirSph(
                _r=v.op_HCM_r,
                _size_tang=v.op_HCM_size_tang,
                _size_sag=v.op_HCM_size_sag,
                _nvx=v.op_HCM_nvx,
                _nvy=v.op_HCM_nvy,
                _nvz=v.op_HCM_nvz,
                _tvx=v.op_HCM_tvx,
                _tvy=v.op_HCM_tvy,
                _x=v.op_HCM_x,
                _y=v.op_HCM_y,
            ))
            pp.append(v.op_HCM_pp)

        elif el_name == 'HCM_Before_DCM':
            # HCM_Before_DCM: drift 28.35m
            el.append(SRWLOptD(
                _L=v.op_HCM_Before_DCM_L,
            ))
            pp.append(v.op_HCM_Before_DCM_pp)
        elif el_name == 'DCM_C1':
            # DCM_C1: crystal 30.42m
            crystal = SRWLOptCryst(
                _d_sp=v.op_DCM_C1_d_sp,
                _psi0r=v.op_DCM_C1_psi0r,
                _psi0i=v.op_DCM_C1_psi0i,
                _psi_hr=v.op_DCM_C1_psiHr,
                _psi_hi=v.op_DCM_C1_psiHi,
                _psi_hbr=v.op_DCM_C1_psiHBr,
                _psi_hbi=v.op_DCM_C1_psiHBi,
                _tc=v.op_DCM_C1_tc,
                _ang_as=v.op_DCM_C1_ang_as,
                _nvx=v.op_DCM_C1_nvx,
                _nvy=v.op_DCM_C1_nvy,
                _nvz=v.op_DCM_C1_nvz,
                _tvx=v.op_DCM_C1_tvx,
                _tvy=v.op_DCM_C1_tvy,
                _uc=v.op_DCM_C1_uc,
                _e_avg=v.op_DCM_C1_energy,
                _ang_roll=v.op_DCM_C1_diffractionAngle
            )
            el.append(crystal)
            pp.append(v.op_DCM_C1_pp)

        elif el_name == 'DCM_C2':
            # DCM_C2: crystal 30.42m
            crystal = SRWLOptCryst(
                _d_sp=v.op_DCM_C2_d_sp,
                _psi0r=v.op_DCM_C2_psi0r,
                _psi0i=v.op_DCM_C2_psi0i,
                _psi_hr=v.op_DCM_C2_psiHr,
                _psi_hi=v.op_DCM_C2_psiHi,
                _psi_hbr=v.op_DCM_C2_psiHBr,
                _psi_hbi=v.op_DCM_C2_psiHBi,
                _tc=v.op_DCM_C2_tc,
                _ang_as=v.op_DCM_C2_ang_as,
                _nvx=v.op_DCM_C2_nvx,
                _nvy=v.op_DCM_C2_nvy,
                _nvz=v.op_DCM_C2_nvz,
                _tvx=v.op_DCM_C2_tvx,
                _tvy=v.op_DCM_C2_tvy,
                _uc=v.op_DCM_C2_uc,
                _e_avg=v.op_DCM_C2_energy,
                _ang_roll=v.op_DCM_C2_diffractionAngle
            )
            el.append(crystal)
            pp.append(v.op_DCM_C2_pp)

        elif el_name == 'After_DCM_HFM':
            # After_DCM_HFM: drift 30.42m
            el.append(SRWLOptD(
                _L=v.op_After_DCM_HFM_L,
            ))
            pp.append(v.op_After_DCM_HFM_pp)
        elif el_name == 'HFM':
            # HFM: sphericalMirror 32.64m
            el.append(SRWLOptMirSph(
                _r=v.op_HFM_r,
                _size_tang=v.op_HFM_size_tang,
                _size_sag=v.op_HFM_size_sag,
                _nvx=v.op_HFM_nvx,
                _nvy=v.op_HFM_nvy,
                _nvz=v.op_HFM_nvz,
                _tvx=v.op_HFM_tvx,
                _tvy=v.op_HFM_tvy,
                _x=v.op_HFM_x,
                _y=v.op_HFM_y,
            ))
            pp.append(v.op_HFM_pp)

        elif el_name == 'HFM_VFM':
            # HFM_VFM: drift 32.64m
            el.append(SRWLOptD(
                _L=v.op_HFM_VFM_L,
            ))
            pp.append(v.op_HFM_VFM_pp)
        elif el_name == 'VFM':
            # VFM: ellipsoidMirror 38.5m
            el.append(SRWLOptMirEl(
                _p=v.op_VFM_p,
                _q=v.op_VFM_q,
                _ang_graz=v.op_VFM_ang,
                _size_tang=v.op_VFM_size_tang,
                _size_sag=v.op_VFM_size_sag,
                _nvx=v.op_VFM_nvx,
                _nvy=v.op_VFM_nvy,
                _nvz=v.op_VFM_nvz,
                _tvx=v.op_VFM_tvx,
                _tvy=v.op_VFM_tvy,
                _x=v.op_VFM_x,
                _y=v.op_VFM_y,
            ))
            pp.append(v.op_VFM_pp)
            mirror_file = v.op_VFM_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by VFM beamline element'.format(mirror_file)
            el.append(srwl_opt_setup_surf_height_1d(
                srwl_uti_read_data_cols(mirror_file, "\t", 0, 1),
                _dim=v.op_VFM_dim,
                _ang=abs(v.op_VFM_ang),
                _amp_coef=v.op_VFM_amp_coef,
            ))
            pp.append([0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0])
        elif el_name == 'VFM_VPM':
            # VFM_VPM: drift 38.5m
            el.append(SRWLOptD(
                _L=v.op_VFM_VPM_L,
            ))
            pp.append(v.op_VFM_VPM_pp)
        elif el_name == 'VPM':
            # VPM: mirror 39.5m
            mirror_file = v.op_VPM_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by VPM beamline element'.format(mirror_file)
            el.append(srwl_opt_setup_surf_height_1d(
                srwl_uti_read_data_cols(mirror_file, "\t", 0, 1),
                _dim=v.op_VPM_dim,
                _ang=abs(v.op_VPM_ang),
                _amp_coef=v.op_VPM_amp_coef,
                _size_x=v.op_VPM_size_x,
                _size_y=v.op_VPM_size_y,
            ))
            pp.append(v.op_VPM_pp)
        elif el_name == 'After_VPM_Before_SSA':
            # After_VPM_Before_SSA: drift 39.5m
            el.append(SRWLOptD(
                _L=v.op_After_VPM_Before_SSA_L,
            ))
            pp.append(v.op_After_VPM_Before_SSA_pp)
        elif el_name == 'SSA':
            # SSA: aperture 94.5m
            el.append(SRWLOptA(
                _shape=v.op_SSA_shape,
                _ap_or_ob='a',
                _Dx=v.op_SSA_Dx,
                _Dy=v.op_SSA_Dy,
                _x=v.op_SSA_x,
                _y=v.op_SSA_y,
            ))
            pp.append(v.op_SSA_pp)
        elif el_name == 'After_SSA_Before_VKB':
            # After_SSA_Before_VKB: drift 94.5m
            el.append(SRWLOptD(
                _L=v.op_After_SSA_Before_VKB_L,
            ))
            pp.append(v.op_After_SSA_Before_VKB_pp)
        elif el_name == 'VKB':
            # VKB: ellipsoidMirror 108.9042143m
            el.append(SRWLOptMirEl(
                _p=v.op_VKB_p,
                _q=v.op_VKB_q,
                _ang_graz=v.op_VKB_ang,
                _size_tang=v.op_VKB_size_tang,
                _size_sag=v.op_VKB_size_sag,
                _nvx=v.op_VKB_nvx,
                _nvy=v.op_VKB_nvy,
                _nvz=v.op_VKB_nvz,
                _tvx=v.op_VKB_tvx,
                _tvy=v.op_VKB_tvy,
                _x=v.op_VKB_x,
                _y=v.op_VKB_y,
            ))
            pp.append(v.op_VKB_pp)

        elif el_name == 'After_VKB_Before_HKB':
            # After_VKB_Before_HKB: drift 108.9042143m
            el.append(SRWLOptD(
                _L=v.op_After_VKB_Before_HKB_L,
            ))
            pp.append(v.op_After_VKB_Before_HKB_pp)
        elif el_name == 'HKB':
            # HKB: ellipsoidMirror 108.9660579m
            el.append(SRWLOptMirEl(
                _p=v.op_HKB_p,
                _q=v.op_HKB_q,
                _ang_graz=v.op_HKB_ang,
                _size_tang=v.op_HKB_size_tang,
                _size_sag=v.op_HKB_size_sag,
                _nvx=v.op_HKB_nvx,
                _nvy=v.op_HKB_nvy,
                _nvz=v.op_HKB_nvz,
                _tvx=v.op_HKB_tvx,
                _tvy=v.op_HKB_tvy,
                _x=v.op_HKB_x,
                _y=v.op_HKB_y,
            ))
            pp.append(v.op_HKB_pp)

        elif el_name == 'After_HKB_Focus':
            # After_HKB_Focus: drift 108.9660579m
            el.append(SRWLOptD(
                _L=v.op_After_HKB_Focus_L,
            ))
            pp.append(v.op_After_HKB_Focus_pp)
        elif el_name == 'Sample':
            # Sample: sample 109.0000025m
            el.append(srwl_opt_setup_transm_from_file(
                file_path=v.op_Sample_file_path,
                resolution=v.op_Sample_resolution,
                thickness=v.op_Sample_thick,
                delta=v.op_Sample_delta,
                atten_len=v.op_Sample_atten_len,
                xc=v.op_Sample_xc,
                yc=v.op_Sample_yc,
                area=None if not v.op_Sample_cropArea else (
                    v.op_Sample_areaXStart,
                    v.op_Sample_areaXEnd,
                    v.op_Sample_areaYStart,
                    v.op_Sample_areaYEnd,
                ),
                extTr=v.op_Sample_extTransm,
                rotate_angle=v.op_Sample_rotateAngle,
                rotate_reshape=bool(int(v.op_Sample_rotateReshape)),
                cutoff_background_noise=v.op_Sample_cutoffBackgroundNoise,
                background_color=v.op_Sample_backgroundColor,
                tile=None if not v.op_Sample_tileImage else (
                    v.op_Sample_tileRows,
                    v.op_Sample_tileColumns,
                ),
                shift_x=v.op_Sample_shiftX,
                shift_y=v.op_Sample_shiftY,
                invert=bool(int(v.op_Sample_invert)),
                is_save_images=False,
                prefix='Sample_sample',
                output_image_format=v.op_Sample_outputImageFormat,
            ))
            pp.append(v.op_Sample_pp)
        elif el_name == 'Watchpoint_Detector':
            # Watchpoint_Detector: drift 109.0000025m
            el.append(SRWLOptD(
                _L=v.op_Watchpoint_Detector_L,
            ))
            pp.append(v.op_Watchpoint_Detector_pp)
    if want_final_propagation:
        pp.append(v.op_fin_pp)

    return SRWLOptC(el, pp)



varParam = [
    ['name', 's', 'NSLS-II HXN Ideal Und. Large Coupl. 12keV KB config1e mono', 'simulation name'],

#---Data Folder
    ['fdir', 's', 'NSLS-II_HXN_Ideal_Und._Large_Coupl._12keV_KB_config1e_mono/', 'folder (directory) name for reading-in input and saving output data files'],

#---Electron Beam
    ['ebm_nm', 's', '', 'standard electron beam name'],
    ['ebm_nms', 's', '', 'standard electron beam name suffix: e.g. can be Day1, Final'],
    ['ebm_i', 'f', 0.5, 'electron beam current [A]'],
    ['ebm_e', 'f', 3.0, 'electron beam avarage energy [GeV]'],
    ['ebm_de', 'f', 0.0, 'electron beam average energy deviation [GeV]'],
    ['ebm_x', 'f', 0.0, 'electron beam initial average horizontal position [m]'],
    ['ebm_y', 'f', 0.0, 'electron beam initial average vertical position [m]'],
    ['ebm_xp', 'f', 0.0, 'electron beam initial average horizontal angle [rad]'],
    ['ebm_yp', 'f', 0.0, 'electron beam initial average vertical angle [rad]'],
    ['ebm_z', 'f', 0., 'electron beam initial average longitudinal position [m]'],
    ['ebm_dr', 'f', -1.8, 'electron beam longitudinal drift [m] to be performed before a required calculation'],
    ['ebm_ens', 'f', 0.00089, 'electron beam relative energy spread'],
    ['ebm_emx', 'f', 7.6e-10, 'electron beam horizontal emittance [m]'],
    ['ebm_emy', 'f', 3e-11, 'electron beam vertical emittance [m]'],
    # Definition of the beam through Twiss:
    ['ebm_betax', 'f', 1.84, 'horizontal beta-function [m]'],
    ['ebm_betay', 'f', 1.17, 'vertical beta-function [m]'],
    ['ebm_alphax', 'f', 0.0, 'horizontal alpha-function [rad]'],
    ['ebm_alphay', 'f', 0.0, 'vertical alpha-function [rad]'],
    ['ebm_etax', 'f', 0.0, 'horizontal dispersion function [m]'],
    ['ebm_etay', 'f', 0.0, 'vertical dispersion function [m]'],
    ['ebm_etaxp', 'f', 0.0, 'horizontal dispersion function derivative [rad]'],
    ['ebm_etayp', 'f', 0.0, 'vertical dispersion function derivative [rad]'],

#---Undulator
#---idealized params
    ['und_bx', 'f', 0.0, 'undulator horizontal peak magnetic field [T]'],
    ['und_by', 'f', 0.92478649, 'undulator vertical peak magnetic field [T]'],
    ['und_phx', 'f', 0.0, 'initial phase of the horizontal magnetic field [rad]'],
    ['und_phy', 'f', 0.0, 'initial phase of the vertical magnetic field [rad]'],
    ['und_sx', 'i', 1, 'undulator horizontal magnetic field symmetry vs longitudinal position'],
    ['und_sy', 'i', -1, 'undulator vertical magnetic field symmetry vs longitudinal position'],
    ['und_b2e', '', '', 'estimate undulator fundamental photon energy (in [eV]) for the amplitude of sinusoidal magnetic field defined by und_b or und_bx, und_by', 'store_true'],
    ['und_e2b', '', '', 'estimate undulator field amplitude (in [T]) for the photon energy defined by w_e', 'store_true'],
#---tabulated params
#    ['und_g', 'f', 5.9127, 'undulator gap [mm] (assumes availability of magnetic measurement or simulation data)'],
#    ['und_ph', 'f', 0.0, 'shift of magnet arrays [mm] for which the field should be set up'],
#    ['und_mfz', 's', 'magn_meas_u20_hxn.zip', 'name of zip-file of directory with magnetic measurement files for different gaps + summary file (if it is defined, it overrides the values of und_mdir and und_mfs)'],
#---both  params
    ['und_zc', 'f', 0.0, 'undulator center longitudinal position [m]'],
    ['und_per', 'f', 0.02, 'undulator period [m]'],
    ['und_len', 'f', 3.0, 'undulator length [m]'],



#---Calculation Types
    # Electron Trajectory
    ['tr', '', '', 'calculate electron trajectory', 'store_true'],
    ['tr_cti', 'f', 0.0, 'initial time moment (c*t) for electron trajectory calculation [m]'],
    ['tr_ctf', 'f', 0.0, 'final time moment (c*t) for electron trajectory calculation [m]'],
    ['tr_np', 'f', 10000, 'number of points for trajectory calculation'],
    ['tr_mag', 'i', 1, 'magnetic field to be used for trajectory calculation: 1- approximate, 2- accurate'],
    ['tr_fn', 's', 'res_trj.dat', 'file name for saving calculated trajectory data'],
    ['tr_pl', 's', '', 'plot the resulting trajectiry in graph(s): ""- dont plot, otherwise the string should list the trajectory components to plot'],

    #Single-Electron Spectrum vs Photon Energy
    ['ss', '', '', 'calculate single-e spectrum vs photon energy', 'store_true'],
    ['ss_ei', 'f', 10000.0, 'initial photon energy [eV] for single-e spectrum vs photon energy calculation'],
    ['ss_ef', 'f', 14000.0, 'final photon energy [eV] for single-e spectrum vs photon energy calculation'],
    ['ss_ne', 'i', 5000, 'number of points vs photon energy for single-e spectrum vs photon energy calculation'],
    ['ss_x', 'f', 0.0, 'horizontal position [m] for single-e spectrum vs photon energy calculation'],
    ['ss_y', 'f', 0.0, 'vertical position [m] for single-e spectrum vs photon energy calculation'],
    ['ss_meth', 'i', 1, 'method to use for single-e spectrum vs photon energy calculation: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
    ['ss_prec', 'f', 0.01, 'relative precision for single-e spectrum vs photon energy calculation (nominal value is 0.01)'],
    ['ss_pol', 'i', 6, 'polarization component to extract after spectrum vs photon energy calculation: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['ss_mag', 'i', 1, 'magnetic field to be used for single-e spectrum vs photon energy calculation: 1- approximate, 2- accurate'],
    ['ss_ft', 's', 'f', 'presentation/domain: "f"- frequency (photon energy), "t"- time'],
    ['ss_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
    ['ss_fn', 's', 'res_spec_se.dat', 'file name for saving calculated single-e spectrum vs photon energy'],
    ['ss_pl', 's', '', 'plot the resulting single-e spectrum in a graph: ""- dont plot, "e"- show plot vs photon energy'],

    #Multi-Electron Spectrum vs Photon Energy (taking into account e-beam emittance, energy spread and collection aperture size)
    ['sm', '', '', 'calculate multi-e spectrum vs photon energy', 'store_true'],
    ['sm_ei', 'f', 100.0, 'initial photon energy [eV] for multi-e spectrum vs photon energy calculation'],
    ['sm_ef', 'f', 20000.0, 'final photon energy [eV] for multi-e spectrum vs photon energy calculation'],
    ['sm_ne', 'i', 10000, 'number of points vs photon energy for multi-e spectrum vs photon energy calculation'],
    ['sm_x', 'f', 0.0, 'horizontal center position [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_rx', 'f', 0.0001, 'range of horizontal position / horizontal aperture size [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_nx', 'i', 1, 'number of points vs horizontal position for multi-e spectrum vs photon energy calculation'],
    ['sm_y', 'f', 0.0, 'vertical center position [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_ry', 'f', 0.0001, 'range of vertical position / vertical aperture size [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_ny', 'i', 1, 'number of points vs vertical position for multi-e spectrum vs photon energy calculation'],
    ['sm_mag', 'i', 1, 'magnetic field to be used for calculation of multi-e spectrum spectrum or intensity distribution: 1- approximate, 2- accurate'],
    ['sm_hi', 'i', 1, 'initial UR spectral harmonic to be taken into account for multi-e spectrum vs photon energy calculation'],
    ['sm_hf', 'i', 15, 'final UR spectral harmonic to be taken into account for multi-e spectrum vs photon energy calculation'],
    ['sm_prl', 'f', 1.0, 'longitudinal integration precision parameter for multi-e spectrum vs photon energy calculation'],
    ['sm_pra', 'f', 1.0, 'azimuthal integration precision parameter for multi-e spectrum vs photon energy calculation'],
    ['sm_meth', 'i', -1, 'method to use for spectrum vs photon energy calculation in case of arbitrary input magnetic field: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler", -1- dont use this accurate integration method (rather use approximate if possible)'],
    ['sm_prec', 'f', 0.01, 'relative precision for spectrum vs photon energy calculation in case of arbitrary input magnetic field (nominal value is 0.01)'],
    ['sm_nm', 'i', 1, 'number of macro-electrons for calculation of spectrum in case of arbitrary input magnetic field'],
    ['sm_na', 'i', 5, 'number of macro-electrons to average on each node at parallel (MPI-based) calculation of spectrum in case of arbitrary input magnetic field'],
    ['sm_ns', 'i', 5, 'saving periodicity (in terms of macro-electrons) for intermediate intensity at calculation of multi-electron spectrum in case of arbitrary input magnetic field'],
    ['sm_type', 'i', 2, 'calculate flux (=1) or flux per unit surface (=2)'],
    ['sm_pol', 'i', 6, 'polarization component to extract after calculation of multi-e flux or intensity: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['sm_rm', 'i', 1, 'method for generation of pseudo-random numbers for e-beam phase-space integration: 1- standard pseudo-random number generator, 2- Halton sequences, 3- LPtau sequences (to be implemented)'],
    ['sm_fn', 's', 'res_spec_me.dat', 'file name for saving calculated milti-e spectrum vs photon energy'],
    ['sm_pl', 's', '', 'plot the resulting spectrum-e spectrum in a graph: ""- dont plot, "e"- show plot vs photon energy'],
    #to add options for the multi-e calculation from "accurate" magnetic field

    #Power Density Distribution vs horizontal and vertical position
    ['pw', '', '', 'calculate SR power density distribution', 'store_true'],
    ['pw_x', 'f', 0.0, 'central horizontal position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_rx', 'f', 0.025, 'range of horizontal position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_nx', 'i', 100, 'number of points vs horizontal position for calculation of power density distribution'],
    ['pw_y', 'f', 0.0, 'central vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_ry', 'f', 0.015, 'range of vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_ny', 'i', 100, 'number of points vs vertical position for calculation of power density distribution'],
    ['pw_pr', 'f', 1.0, 'precision factor for calculation of power density distribution'],
    ['pw_meth', 'i', 1, 'power density computation method (1- "near field", 2- "far field")'],
    ['pw_zst', 'f', 0., 'initial longitudinal position along electron trajectory of power density distribution (effective if pow_sst < pow_sfi)'],
    ['pw_zfi', 'f', 0., 'final longitudinal position along electron trajectory of power density distribution (effective if pow_sst < pow_sfi)'],
    ['pw_mag', 'i', 1, 'magnetic field to be used for power density calculation: 1- approximate, 2- accurate'],
    ['pw_fn', 's', 'res_pow.dat', 'file name for saving calculated power density distribution'],
    ['pw_pl', 's', '', 'plot the resulting power density distribution in a graph: ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

    #Single-Electron Intensity distribution vs horizontal and vertical position
    ['si', '', '', 'calculate single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position', 'store_true'],
    #Single-Electron Wavefront Propagation
    ['ws', '', '', 'calculate single-electron (/ fully coherent) wavefront propagation', 'store_true'],
    #Multi-Electron (partially-coherent) Wavefront Propagation
    ['wm', '', '', 'calculate multi-electron (/ partially coherent) wavefront propagation', 'store_true'],

    ['w_e', 'f', 12000.0, 'photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ef', 'f', -1.0, 'final photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ne', 'i', 1, 'number of points vs photon energy for calculation of intensity distribution'],
    ['w_x', 'f', 0.0, 'central horizontal position [m] for calculation of intensity distribution'],
    ['w_rx', 'f', 0.0025, 'range of horizontal position [m] for calculation of intensity distribution'],
    ['w_nx', 'i', 96, 'number of points vs horizontal position for calculation of intensity distribution'],
    ['w_y', 'f', 0.0, 'central vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ry', 'f', 0.0015, 'range of vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ny', 'i', 60, 'number of points vs vertical position for calculation of intensity distribution'],
    ['w_smpf', 'f', 0.05, 'sampling factor for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_meth', 'i', 1, 'method to use for calculation of intensity distribution vs horizontal and vertical position: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
    ['w_prec', 'f', 0.01, 'relative precision for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_mag', 'i', 1, 'magnetic field to be used for calculation of intensity distribution vs horizontal and vertical position: 1- approximate, 2- accurate'],
    ['w_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],

    ['si_pol', 'i', 6, 'polarization component to extract after calculation of intensity distribution: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['si_type', 'i', 0, 'type of a characteristic to be extracted after calculation of intensity distribution: 0- Single-Electron Intensity, 1- Multi-Electron Intensity, 2- Single-Electron Flux, 3- Multi-Electron Flux, 4- Single-Electron Radiation Phase, 5- Re(E): Real part of Single-Electron Electric Field, 6- Im(E): Imaginary part of Single-Electron Electric Field, 7- Single-Electron Intensity, integrated over Time or Photon Energy'],
    ['si_fn', 's', 'res_int_se.dat', 'file name for saving calculated single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position'],
    ['si_pl', 's', '', 'plot the input intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],
    ['ws_fni', 's', 'res_int_pr_se.dat', 'file name for saving propagated single-e intensity distribution vs horizontal and vertical position'],
    ['ws_pl', 's', '', 'plot the resulting intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

    ['wm_nm', 'i', 1000000, 'number of macro-electrons (coherent wavefronts) for calculation of multi-electron wavefront propagation'],
    ['wm_na', 'i', 5, 'number of macro-electrons (coherent wavefronts) to average on each node for parallel (MPI-based) calculation of multi-electron wavefront propagation'],
    ['wm_ns', 'i', 5, 'saving periodicity (in terms of macro-electrons / coherent wavefronts) for intermediate intensity at multi-electron wavefront propagation calculation'],
    ['wm_ch', 'i', 0, 'type of a characteristic to be extracted after calculation of multi-electron wavefront propagation: #0- intensity (s0); 1- four Stokes components; 2- mutual intensity cut vs x; 3- mutual intensity cut vs y; 40- intensity(s0), mutual intensity cuts and degree of coherence vs X & Y'],
    ['wm_ap', 'i', 0, 'switch specifying representation of the resulting Stokes parameters: coordinate (0) or angular (1)'],
    ['wm_x0', 'f', 0.0, 'horizontal center position for mutual intensity cut calculation'],
    ['wm_y0', 'f', 0.0, 'vertical center position for mutual intensity cut calculation'],
    ['wm_ei', 'i', 0, 'integration over photon energy is required (1) or not (0); if the integration is required, the limits are taken from w_e, w_ef'],
    ['wm_rm', 'i', 1, 'method for generation of pseudo-random numbers for e-beam phase-space integration: 1- standard pseudo-random number generator, 2- Halton sequences, 3- LPtau sequences (to be implemented)'],
    ['wm_am', 'i', 0, 'multi-electron integration approximation method: 0- no approximation (use the standard 5D integration method), 1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance'],
    ['wm_fni', 's', 'res_int_pr_me.dat', 'file name for saving propagated multi-e intensity distribution vs horizontal and vertical position'],
    ['wm_ff', 's', 'ascii', 'format of file name for saving propagated multi-e intensity distribution vs horizontal and vertical position (ascii and hdf5 supported)'],

    ['wm_nmm', 'i', 1, 'number of MPI masters to use'],
    ['wm_ncm', 'i', 100, 'number of Coherent Modes to calculate'],
    ['wm_acm', 's', 'SP', 'coherent mode decomposition algorithm to be used (supported algorithms are: "SP" for SciPy, "SPS" for SciPy Sparse, "PM" for Primme, based on names of software packages)'],
    ['wm_nop', '', '', 'switch forcing to do calculations ignoring any optics defined (by set_optics function)', 'store_true'],

    ['wm_fnmi', 's', '', 'file name of input cross-spectral density / mutual intensity; if this file name is supplied, the initial cross-spectral density (for such operations as coherent mode decomposition) will not be calculated, but rathre it will be taken from that file.'],
    ['wm_fncm', 's', '', 'file name of input coherent modes; if this file name is supplied, the eventual partially-coherent radiation propagation simulation will be done based on propagation of the coherent modes from that file.'],

    ['wm_fbk', '', '', 'create backup file(s) with propagated multi-e intensity distribution vs horizontal and vertical position and other radiation characteristics', 'store_true'],

    # Optics parameters
    ['op_r', 'f', 26.62, 'longitudinal position of the first optical element [m]'],
    # Former appParam:
    ['rs_type', 's', 'u', 'source type, (u) idealized undulator, (t), tabulated undulator, (m) multipole, (g) gaussian beam'],

#---Beamline optics:
    # S1: aperture
    ['op_S1_shape', 's', 'r', 'shape'],
    ['op_S1_Dx', 'f', 0.0025, 'horizontalSize'],
    ['op_S1_Dy', 'f', 0.0007, 'verticalSize'],
    ['op_S1_x', 'f', 0.0, 'horizontalOffset'],
    ['op_S1_y', 'f', 0.0, 'verticalOffset'],

    # S1_HCM: drift
    ['op_S1_HCM_L', 'f', 1.7300000000000004, 'length'],

    # HCM: sphericalMirror
    ['op_HCM_hfn', 's', '', 'heightProfileFile'],
    ['op_HCM_dim', 's', 'x', 'orientation'],
    ['op_HCM_r', 'f', 17718.8, 'radius'],
    ['op_HCM_size_tang', 'f', 1.0, 'tangentialSize'],
    ['op_HCM_size_sag', 'f', 0.006, 'sagittalSize'],
    ['op_HCM_ang', 'f', 0.0032, 'grazingAngle'],
    ['op_HCM_nvx', 'f', 0.9999948800043691, 'normalVectorX'],
    ['op_HCM_nvy', 'f', 0.0, 'normalVectorY'],
    ['op_HCM_nvz', 'f', -0.003199994538669463, 'normalVectorZ'],
    ['op_HCM_tvx', 'f', 0.003199994538669463, 'tangentialVectorX'],
    ['op_HCM_tvy', 'f', 0.0, 'tangentialVectorY'],
    ['op_HCM_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_HCM_x', 'f', 0.0, 'horizontalOffset'],
    ['op_HCM_y', 'f', 0.0, 'verticalOffset'],

    # HCM_Before_DCM: drift
    ['op_HCM_Before_DCM_L', 'f', 2.0700000000000003, 'length'],

    # DCM_C1: crystal
    ['op_DCM_C1_hfn', 's', '', 'heightProfileFile'],
    ['op_DCM_C1_dim', 's', 'x', 'orientation'],
    ['op_DCM_C1_d_sp', 'f', 3.1355713563754857, 'dSpacing'],
    ['op_DCM_C1_psi0r', 'f', -6.759642582545276e-06, 'psi0r'],
    ['op_DCM_C1_psi0i', 'f', 7.245217017493692e-08, 'psi0i'],
    ['op_DCM_C1_psiHr', 'f', -3.56789300932007e-06, 'psiHr'],
    ['op_DCM_C1_psiHi', 'f', 5.058419598704464e-08, 'psiHi'],
    ['op_DCM_C1_psiHBr', 'f', -3.56789300932007e-06, 'psiHBr'],
    ['op_DCM_C1_psiHBi', 'f', 5.058419598704464e-08, 'psiHBi'],
    ['op_DCM_C1_tc', 'f', 0.01, 'crystalThickness'],
    ['op_DCM_C1_uc', 'f', 1, 'useCase'],
    ['op_DCM_C1_ang_as', 'f', 0.0, 'asymmetryAngle'],
    ['op_DCM_C1_nvx', 'f', -0.9863311088533899, 'nvx'],
    ['op_DCM_C1_nvy', 'f', 6.702018003143382e-09, 'nvy'],
    ['op_DCM_C1_nvz', 'f', -0.16477543417646437, 'nvz'],
    ['op_DCM_C1_tvx', 'f', -0.16477543417646437, 'tvx'],
    ['op_DCM_C1_tvy', 'f', 1.119632055010627e-09, 'tvy'],
    ['op_DCM_C1_ang', 'f', 0.1655303290490928, 'grazingAngle'],
    ['op_DCM_C1_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_DCM_C1_energy', 'f', 12000.0, 'energy'],
    ['op_DCM_C1_diffractionAngle', 'f', 1.57079632, 'diffractionAngle'],

    # DCM_C2: crystal
    ['op_DCM_C2_hfn', 's', '', 'heightProfileFile'],
    ['op_DCM_C2_dim', 's', 'x', 'orientation'],
    ['op_DCM_C2_d_sp', 'f', 3.1355713563754857, 'dSpacing'],
    ['op_DCM_C2_psi0r', 'f', -6.759642582545276e-06, 'psi0r'],
    ['op_DCM_C2_psi0i', 'f', 7.245217017493692e-08, 'psi0i'],
    ['op_DCM_C2_psiHr', 'f', -3.56789300932007e-06, 'psiHr'],
    ['op_DCM_C2_psiHi', 'f', 5.058419598704464e-08, 'psiHi'],
    ['op_DCM_C2_psiHBr', 'f', -3.56789300932007e-06, 'psiHBr'],
    ['op_DCM_C2_psiHBi', 'f', 5.058419598704464e-08, 'psiHBi'],
    ['op_DCM_C2_tc', 'f', 0.01, 'crystalThickness'],
    ['op_DCM_C2_uc', 'f', 1, 'useCase'],
    ['op_DCM_C2_ang_as', 'f', 0.0, 'asymmetryAngle'],
    ['op_DCM_C2_nvx', 'f', 0.9863311088533899, 'nvx'],
    ['op_DCM_C2_nvy', 'f', 6.702018003143382e-09, 'nvy'],
    ['op_DCM_C2_nvz', 'f', -0.16477543417646437, 'nvz'],
    ['op_DCM_C2_tvx', 'f', 0.16477543417646437, 'tvx'],
    ['op_DCM_C2_tvy', 'f', 1.119632055010627e-09, 'tvy'],
    ['op_DCM_C2_ang', 'f', 0.1655303290490928, 'grazingAngle'],
    ['op_DCM_C2_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_DCM_C2_energy', 'f', 12000.0, 'energy'],
    ['op_DCM_C2_diffractionAngle', 'f', -1.57079632, 'diffractionAngle'],

    # After_DCM_HFM: drift
    ['op_After_DCM_HFM_L', 'f', 2.219999999999999, 'length'],

    # HFM: sphericalMirror
    ['op_HFM_hfn', 's', '', 'heightProfileFile'],
    ['op_HFM_dim', 's', 'x', 'orientation'],
    ['op_HFM_r', 'f', 38660.0, 'radius'],
    ['op_HFM_size_tang', 'f', 1.0, 'tangentialSize'],
    ['op_HFM_size_sag', 'f', 0.06, 'sagittalSize'],
    ['op_HFM_ang', 'f', 0.0032, 'grazingAngle'],
    ['op_HFM_nvx', 'f', -0.9999948800043691, 'normalVectorX'],
    ['op_HFM_nvy', 'f', 0.0, 'normalVectorY'],
    ['op_HFM_nvz', 'f', -0.003199994538669463, 'normalVectorZ'],
    ['op_HFM_tvx', 'f', -0.003199994538669463, 'tangentialVectorX'],
    ['op_HFM_tvy', 'f', 0.0, 'tangentialVectorY'],
    ['op_HFM_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_HFM_x', 'f', 0.0, 'horizontalOffset'],
    ['op_HFM_y', 'f', 0.0, 'verticalOffset'],

    # HFM_VFM: drift
    ['op_HFM_VFM_L', 'f', 5.859999999999999, 'length'],

    # VFM: ellipsoidMirror
    ['op_VFM_hfn', 's', 'NSLS-II_HXN_Ideal_Und._Large_Coupl._12keV_KB_config1e_mono/hxn_vfm_figure_error_1207.dat', 'heightProfileFile'],
    ['op_VFM_dim', 's', 'y', 'orientation'],
    ['op_VFM_p', 'f', 38.5, 'firstFocusLength'],
    ['op_VFM_q', 'f', 55.81, 'focalLength'],
    ['op_VFM_ang', 'f', 0.003, 'grazingAngle'],
    ['op_VFM_amp_coef', 'f', 2.0, 'heightAmplification'],
    ['op_VFM_size_tang', 'f', 0.45, 'tangentialSize'],
    ['op_VFM_size_sag', 'f', 0.01, 'sagittalSize'],
    ['op_VFM_nvx', 'f', 0.0, 'normalVectorX'],
    ['op_VFM_nvy', 'f', 0.999995500003375, 'normalVectorY'],
    ['op_VFM_nvz', 'f', -0.002999995500002025, 'normalVectorZ'],
    ['op_VFM_tvx', 'f', 0.0, 'tangentialVectorX'],
    ['op_VFM_tvy', 'f', 0.002999995500002025, 'tangentialVectorY'],
    ['op_VFM_x', 'f', 0.0, 'horizontalOffset'],
    ['op_VFM_y', 'f', 0.0, 'verticalOffset'],

    # VFM_VPM: drift
    ['op_VFM_VPM_L', 'f', 1.0, 'length'],

    # VPM: mirror
    ['op_VPM_hfn', 's', 'NSLS-II_HXN_Ideal_Und._Large_Coupl._12keV_KB_config1e_mono/mirror_1d.dat', 'heightProfileFile'],
    ['op_VPM_dim', 's', 'y', 'orientation'],
    ['op_VPM_ang', 'f', 0.003, 'grazingAngle'],
    ['op_VPM_amp_coef', 'f', 0.0, 'heightAmplification'],
    ['op_VPM_size_x', 'f', 0.001, 'horizontalTransverseSize'],
    ['op_VPM_size_y', 'f', 0.001, 'verticalTransverseSize'],

    # After_VPM_Before_SSA: drift
    ['op_After_VPM_Before_SSA_L', 'f', 55.0, 'length'],

    # SSA: aperture
    ['op_SSA_shape', 's', 'r', 'shape'],
    ['op_SSA_Dx', 'f', 1.7e-05, 'horizontalSize'],
    ['op_SSA_Dy', 'f', 2e-05, 'verticalSize'],
    ['op_SSA_x', 'f', 0.0, 'horizontalOffset'],
    ['op_SSA_y', 'f', 0.0, 'verticalOffset'],

    # After_SSA_Before_VKB: drift
    ['op_After_SSA_Before_VKB_L', 'f', 14.404214300000007, 'length'],

    # VKB: ellipsoidMirror
    ['op_VKB_hfn', 's', '', 'heightProfileFile'],
    ['op_VKB_dim', 's', 'y', 'orientation'],
    ['op_VKB_p', 'f', 14.4042143, 'firstFocusLength'],
    ['op_VKB_q', 'f', 0.0957882, 'focalLength'],
    ['op_VKB_ang', 'f', 0.0033, 'grazingAngle'],
    ['op_VKB_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_VKB_size_tang', 'f', 0.0616, 'tangentialSize'],
    ['op_VKB_size_sag', 'f', 0.01, 'sagittalSize'],
    ['op_VKB_nvx', 'f', 0.0, 'normalVectorX'],
    ['op_VKB_nvy', 'f', 0.9999945550049414, 'normalVectorY'],
    ['op_VKB_nvz', 'f', -0.003299994010503261, 'normalVectorZ'],
    ['op_VKB_tvx', 'f', 0.0, 'tangentialVectorX'],
    ['op_VKB_tvy', 'f', 0.003299994010503261, 'tangentialVectorY'],
    ['op_VKB_x', 'f', 0.0, 'horizontalOffset'],
    ['op_VKB_y', 'f', 0.0, 'verticalOffset'],

    # After_VKB_Before_HKB: drift
    ['op_After_VKB_Before_HKB_L', 'f', 0.061843599999988896, 'length'],

    # HKB: ellipsoidMirror
    ['op_HKB_hfn', 's', '', 'heightProfileFile'],
    ['op_HKB_dim', 's', 'x', 'orientation'],
    ['op_HKB_p', 'f', 14.4660579, 'firstFocusLength'],
    ['op_HKB_q', 'f', 0.0339446, 'focalLength'],
    ['op_HKB_ang', 'f', 0.0024, 'grazingAngle'],
    ['op_HKB_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_HKB_size_tang', 'f', 0.0439, 'tangentialSize'],
    ['op_HKB_size_sag', 'f', 0.01, 'sagittalSize'],
    ['op_HKB_nvx', 'f', 0.9999971200013824, 'normalVectorX'],
    ['op_HKB_nvy', 'f', 0.0, 'normalVectorY'],
    ['op_HKB_nvz', 'f', -0.0023999976960006634, 'normalVectorZ'],
    ['op_HKB_tvx', 'f', -0.0023999976960006634, 'tangentialVectorX'],
    ['op_HKB_tvy', 'f', 0.0, 'tangentialVectorY'],
    ['op_HKB_x', 'f', 0.0, 'horizontalOffset'],
    ['op_HKB_y', 'f', 0.0, 'verticalOffset'],

    # After_HKB_Focus: drift
    ['op_After_HKB_Focus_L', 'f', 0.033944599999998104, 'length'],

    # Sample: sample
    ['op_Sample_file_path', 's', 'NSLS-II_HXN_Ideal_Und._Large_Coupl._12keV_KB_config1e_mono/Siemens_star_128_spokes__Matlab_code.svg.png', 'imageFile'],
    ['op_Sample_outputImageFormat', 's', 'tif', 'outputImageFormat'],
    ['op_Sample_position', 'f', 109.0000025, 'position'],
    ['op_Sample_resolution', 'f', 1e-9, 'resolution'],
    ['op_Sample_thick', 'f', 2e-06, 'thickness'],
    ['op_Sample_delta', 'f', 1.840189e-05, 'refractiveIndex'],
    ['op_Sample_atten_len', 'f', 2.88087e-06, 'attenuationLength'],
    ['op_Sample_xc', 'f', 1e-08, 'horizontalCenterCoordinate'],
    ['op_Sample_yc', 'f', 1e-08, 'verticalCenterCoordinate'],
    ['op_Sample_rotateAngle', 'f', 0.0, 'rotateAngle'],
    ['op_Sample_cutoffBackgroundNoise', 'f', 0.0, 'cutoffBackgroundNoise'],
    ['op_Sample_rx', 'f', 1e-05, 'rx'],
    ['op_Sample_ry', 'f', 1e-05, 'ry'],
    ['op_Sample_dens', 'f', 20000000.0, 'dens'],
    ['op_Sample_r_min_bw_obj', 'f', 1e-09, 'r_min_bw_obj'],
    ['op_Sample_edge_frac', 'f', 0.02, 'edge_frac'],
    ['op_Sample_obj_size_min', 'f', 1e-07, 'obj_size_min'],
    ['op_Sample_ang_min', 'f', 0.0, 'ang_min'],
    ['op_Sample_obj_size_max', 'f', 1.2e-07, 'obj_size_max'],
    ['op_Sample_ang_max', 'f', 45.0, 'ang_max'],
    ['op_Sample_obj_size_ratio', 'f', 0.5, 'obj_size_ratio'],
    ['op_Sample_cropArea', 'i', 0, 'cropArea'],
    ['op_Sample_extTransm', 'i', 1, 'transmissionImage'],
    ['op_Sample_areaXStart', 'i', 0, 'areaXStart'],
    ['op_Sample_areaXEnd', 'i', 1280, 'areaXEnd'],
    ['op_Sample_areaYStart', 'i', 0, 'areaYStart'],
    ['op_Sample_areaYEnd', 'i', 834, 'areaYEnd'],
    ['op_Sample_rotateReshape', 'i', 0, 'rotateReshape'],
    ['op_Sample_backgroundColor', 'i', 0, 'backgroundColor'],
    ['op_Sample_tileImage', 'i', 0, 'tileImage'],
    ['op_Sample_tileRows', 'i', 1, 'tileRows'],
    ['op_Sample_tileColumns', 'i', 1, 'tileColumns'],
    ['op_Sample_shiftX', 'i', 0, 'shiftX'],
    ['op_Sample_shiftY', 'i', 0, 'shiftY'],
    ['op_Sample_invert', 'i', 0, 'invert'],
    ['op_Sample_nx', 'i', 1001, 'nx'],
    ['op_Sample_ny', 'i', 1001, 'ny'],
    ['op_Sample_obj_type', 'i', 1, 'obj_type'],
    ['op_Sample_size_dist', 'i', 1, 'size_dist'],
    ['op_Sample_ang_dist', 'i', 1, 'ang_dist'],
    ['op_Sample_rand_alg', 'i', 1, 'rand_alg'],
    ['op_Sample_poly_sides', 'i', 6, 'poly_sides'],
    ['op_Sample_rand_shapes', 'i', [1, 2, 3, 4], 'rand_shapes'],
    ['op_Sample_rand_obj_size', 'b', False, 'rand_obj_size'],
    ['op_Sample_rand_poly_side', 'b', False, 'rand_poly_side'],

    # Watchpoint_Detector: drift
    ['op_Watchpoint_Detector_L', 'f', 0.4, 'length'],

#---Propagation parameters
    ['op_S1_pp', 'f',                   [0, 0, 1.0, 0, 0, 1.2, 1.4, 1.0, 6.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'S1'],
    ['op_S1_HCM_pp', 'f',               [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'S1_HCM'],
    ['op_HCM_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'HCM'],
    ['op_HCM_Before_DCM_pp', 'f',       [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'HCM_Before_DCM'],
    ['op_DCM_C1_pp', 'f',               [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.32504627340614156, 2.208655852652862e-09, 0.9456981125839153, 0.9456981125839153, 3.689757159645261e-10], 'DCM_C1'],
    ['op_DCM_C2_pp', 'f',               [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.32504627340614156, 2.208655852652862e-09, 0.9456981125839153, 0.9456981125839153, -3.689757159645261e-10], 'DCM_C2'],
    ['op_After_DCM_HFM_pp', 'f',        [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_DCM_HFM'],
    ['op_HFM_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'HFM'],
    ['op_HFM_VFM_pp', 'f',              [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'HFM_VFM'],
    ['op_VFM_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'VFM'],
    ['op_VFM_VPM_pp', 'f',              [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'VFM_VPM'],
    ['op_VPM_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'VPM'],
    ['op_After_VPM_Before_SSA_pp', 'f', [0, 0, 1.0, 4, 0, 2.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_VPM_Before_SSA'],
    ['op_SSA_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'SSA'],
    ['op_After_SSA_Before_VKB_pp', 'f', [0, 0, 1.0, 3, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_SSA_Before_VKB'],
    ['op_VKB_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'VKB'],
    ['op_After_VKB_Before_HKB_pp', 'f', [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_VKB_Before_HKB'],
    ['op_HKB_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'HKB'],
    ['op_After_HKB_Focus_pp', 'f',      [0, 0, 1.0, 4, 0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_HKB_Focus'],
    ['op_Sample_pp', 'f',               [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Sample'],
    ['op_Watchpoint_Detector_pp', 'f',  [0, 0, 1.0, 3, 0, 4.0, 1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Watchpoint_Detector'],
    ['op_fin_pp', 'f',                  [0, 0, 1.0, 0, 0, 0.1, 1.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'final post-propagation (resize) parameters'],

    #[ 0]: Auto-Resize (1) or not (0) Before propagation
    #[ 1]: Auto-Resize (1) or not (0) After propagation
    #[ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
    #[ 3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
    #[ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
    #[ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
    #[ 6]: Horizontal Resolution modification factor at Resizing
    #[ 7]: Vertical Range modification factor at Resizing
    #[ 8]: Vertical Resolution modification factor at Resizing
    #[ 9]: Type of wavefront Shift before Resizing (not yet implemented)
    #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
    #[11]: New Vertical wavefront Center position after Shift (not yet implemented)
    #[12]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Horizontal Coordinate
    #[13]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Vertical Coordinate
    #[14]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Longitudinal Coordinate
    #[15]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Horizontal Coordinate
    #[16]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Vertical Coordinate

    ['scan_mode', 'i', 0, '0=scanning, -1=generating ptyd file, -2=ptychographic reconstruction'],
    ['scan_width', 'f', 1e-06, 'scan width [m]'],
    ['scan_height', 'f', 1e-06, 'scan height [m]'],
    ['scan_step', 'f', 0.05e-06, 'scan step size [m]'],
    ['scan_jitter', 'i', 0, '0=disable, 1=enable, Apply jitter to the spectrum and transverse position of the pulse'],
    ['scan_det_noise', 'i', 1, '0=disable, 1=enable, Apply photon noise to the detector'],
    ['scatter_calc', 'i', 1, '0=disable, 1=enable, Calculate the scattered intensity'],
]

#------------------------------------------------------------------------------
#def epilogue():
#    pass
    
def fft2(img, norm = 'forward'):
    import scipy.fft
    #ic = cp.array(img, dtype=np.complex64)
    #return cp.fft.fft2(ic, norm=norm).get()
    return scipy.fft.fft2(img, norm=norm)
    cufft_type = cp.cuda.cufft.CUFFT_C2C
    plan = cp.cuda.cufft.Plan2d(img.shape[0], img.shape[1], cufft_type, batch=1, devices=[0, 1, 2, 3])
    out_cp = cp.zeros(img.shape, dtype=np.complex64)
    plan.fft(img, out_cp, cp.cuda.cufft.CUFFT_FORWARD)
    return out_cp

def ifft2(img, norm = 'forward'):
    import scipy.fft
    #ic = cp.array(img, dtype=np.complex64)
    #return cp.fft.ifft2(ic, norm=norm).get()
    return scipy.fft.ifft2(img, norm=norm)
    cufft_type = cp.cuda.cufft.CUFFT_C2C
    plan = cp.cuda.cufft.Plan2d(img.shape[0], img.shape[1], cufft_type, batch=1, devices=[0, 1, 2, 3])
    out_cp = cp.zeros(img.shape, dtype=np.complex64)
    plan.ifft(img, out_cp, cp.cuda.cufft.CUFFT_INVERSE)
    return out_cp

def fourier_upsample(img, resolution, dim = None, factor = 1):
    if dim is None and factor == 1:
        return img, resolution
    elif dim is not None and factor == 1:
        if dim[0] == img.shape[0] and dim[1] == img.shape[1]:
            return img, resolution

    img = fft2(img, norm='forward')
    img = np.fft.fftshift(img)
    
    if dim is None:
        dim = [[(img.shape[0] * factor)//2, (img.shape[0] * factor)//2], [(img.shape[1] * factor)//2, (img.shape[1] * factor)//2]]
    else:
        dim = [np.round(dim[0]).astype(int), np.round(dim[1]).astype(int)]
        dim = [[(dim[0] * factor)//2, (dim[0] * factor)//2], [(dim[1] * factor)//2, (dim[1] * factor)//2]]
    
    dim = [[dim[0][0] - img.shape[0] // 2, dim[0][1] - img.shape[0] // 2], [dim[1][0] - img.shape[1] // 2, dim[1][1] - img.shape[1] // 2]]

    for i in range(2):
        for j in range(2):
            dim[i][j] = int(dim[i][j])

    img = np.pad(img, dim, mode='constant')
    img = np.fft.ifftshift(img)
    img = ifft2(img, norm='forward')
    return img, [resolution[0] / factor, resolution[1] / factor]

def spiral_pattern(scanWidth, scanHeight, scanStep, pixelSize):
    #scanWidth and scanHeight are the dimensions of the scan in pixels
    #scanStep is the distance between each point in the scan in microns
    #pixelSize is the size of each pixel in microns
    #Returns a list of points in the spiral pattern

    #The spiral pattern is generated by starting at the center of the scan area and spiraling outward
    scanWidth_px = math.ceil(scanWidth / pixelSize)
    scanHeight_px = math.ceil(scanHeight / pixelSize)
    scanStep_px = math.ceil(scanStep / pixelSize)

    numSteps_x = math.ceil(scanWidth_px / scanStep_px)
    numSteps_y = math.ceil(scanHeight_px / scanStep_px)

    numCircles = max(numSteps_x, numSteps_y)
    spiralPoints = []
    for i in range(numCircles):
        rad = i * scanStep
        if rad == 0:
            spiralPoints.append((0, 0))
            continue

        numPoints = int(2 * math.pi * rad / scanStep)
        angleStep = 2 * math.pi / numPoints
        for j in range(numPoints):
            x = rad * math.cos(j * angleStep)
            y = rad * math.sin(j * angleStep)
            if x < -scanWidth / 2 or x > scanWidth / 2 or y < -scanHeight / 2 or y > scanHeight / 2:
                continue
            spiralPoints.append((x, y))
    return spiralPoints

def convert_to_ptyd(spiralSteps, energy_lvl, sample_to_det_dist, px_sz, sample_resolution, scatterfile_template, ptyd_fname, info_template=None, frame_sz = (1024, 1024), frame_center = (512, 512), mask = None):
    import numpy as np
    try:
        import srwlib
    except ImportError:
        import srwlpy.srwlib as srwlib
    from ptypy.core.data import PtyScan
    from ptypy import utils as u
    from skimage.filters import window as win

    class PtychoScan(PtyScan):
        """
        A PtyScan subclass to extract data from Ptycho style h5 files.

        defaults:

        [name]
        type = str
        default = ptychoscan
        help =

        [base_path]
        type = str
        default = './scan_01.h5'
        help = File path for the scan

        [auto_center]
        default = False

        [dfile]
        default = 'scan_01.ptyd'
        """

        def __init__(self, pars=None, **kwargs):
            p = self.DEFAULT.copy(depth=2)
            p.update(pars)

            p.dfile = ptyd_fname + '.ptyd'
            #p.auto_center = True
            p['name'] = ptyd_fname
            
            p['energy'] = energy_lvl / 1000
            p['distance'] = sample_to_det_dist
            p['psize'] = px_sz
            p['resolution'] = sample_resolution
            #p['shape'] = frame_sz
            #p['shape'] = (frame_center[0] * 2, frame_center[1] * 2)
            p['shape'] = (1411, 1411)
            p['center'] = frame_center
            p['orientation'] = 5
            #p['num_frames'] = 400 #len(spiralSteps)

            raw = {}
            weights = {}
            positions = {}

            if mask is None:
                base_mask = None
            else:
                base_mask = mask

            
            steps = np.array(spiralSteps).T
            steps[0] -= np.min(steps[0])
            steps[1] -= np.min(steps[1])
            steps = steps.T

            filter_window = win('tukey', p['shape'])

            full_sum = 0
            full_max = 0
            index_offset = 0
            for ii in range(len(spiralSteps)):
                norm_factor = 1
                #if info_template is not None:
                #    norm_factor = 1. / np.loadtxt(info_template % ii, usecols=0, delimiter=',')[2]
                #probe, _, _, _ = srwlib.srwl_uti_read_intens_hdf5(info_template % ii)
                #norm_factor = 1. / np.array(probe).sum()

                pos = steps[ii]

                while True:
                    if not os.path.exists(scatterfile_template % (ii + index_offset)):
                        index_offset = -1
                        break
                    data, mesh, _, _ = srwlib.srwl_uti_read_intens_hdf5(scatterfile_template % (ii + index_offset))
                    tot_ph = np.sum(data)
                    cur_pos = steps[ii + index_offset]

                    if tot_ph < 4e7 * 0.2:
                    #if (ii+index_offset) % 2 != 0:
                        print('Skipping %d (Ph = %e)' % (ii + index_offset, tot_ph))
                        raw_tmp = np.array(data).reshape((mesh.ny, mesh.nx)) * norm_factor
                        plt.imsave("raw_%d.png" % (ii + index_offset), np.log10(raw_tmp), vmin = 0, vmax = np.log10(np.max(raw_tmp)))
                        index_offset += 1
                    else:
                        raw_tmp = np.array(data).reshape((mesh.ny, mesh.nx)) * norm_factor
                        #raw_tmp = raw_tmp[:frame_center[0]*2,:frame_center[1]*2] * filter_window
                        #print ("Total Ph: %e" % (tot_ph))
                        plt.imsave("raw_%d.png" % (ii + index_offset), np.log10(raw_tmp), vmin = 0, vmax = np.log10(np.max(raw_tmp)))
                        raw_tmp /= tot_ph
                        raw[ii] = raw_tmp
                        if base_mask is None:
                            base_mask = np.ones_like(raw[ii])
                        weights[ii] = base_mask
                        #weights[ii] = base_mask[:frame_center[0]*2,:frame_center[1]*2]
                        positions[ii] = cur_pos
                        break
                
                if index_offset == -1:
                    break
            print(len(positions))
            p['num_frames'] = len(positions)
            
            self.positions = positions
            self.raw = raw
            self.weights = weights

            super(PtychoScan, self).__init__(p, **kwargs)

        #def load_positions(self):
        #    return self.positions

        def load(self, indices):
            print("loading...")
            raw = {}
            positions = {}
            weights = {}

            for ii in indices:
                raw[ii] = self.raw[ii]
                positions[ii] = self.positions[ii]
                weights[ii] = self.weights[ii]

            return raw, positions, weights

    NPS = PtychoScan()
    NPS.initialize()

    data = NPS.DEFAULT.copy(depth=2)
    data.save = 'append'
    NPS = PtychoScan(pars=data)
    NPS.initialize()
    msg = NPS.auto(len(spiralSteps))
    print(u.verbose.report(NPS, noheader=True))

def ptypy_run(run_name, sample_name, plot_imgs = True, iter_count = 30000, probe_mode_cnt=1, probe_def=None, obj_mode_cnt=1):
    from ptypy.core import Ptycho
    from ptypy import utils as u
    import os
    import ptypy
    ptypy.load_gpu_engines(arch="cupy")

    import tempfile
    tmpdir = tempfile.gettempdir()

    p = u.Param()
    p.verbose_level = "info"
    p.io = u.Param()
    p.io.home = "/".join(['.', run_name, sample_name + "_recons"])
    os.makedirs(p.io.home, exist_ok=True)

    p.io.autoplot = u.Param()
    if plot_imgs:
        p.io.autoplot.active = True
        p.io.autoplot.imfile = "/" + os.path.join(sample_name + "_recons_img", "%(iterations)05d.png")
        #p.io.autoplot.interval = 1
        p.io.autoplot.layout = 'black_and_white'
        p.io.autoplot.make_movie = True
        p.io.autoplot.threaded = True
        p.io.autoplot.dump = True
    else:
        p.io.autoplot.active = False

    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.data= u.Param()
    p.scans.MF.name = 'BlockFull'
    p.scans.MF.data.name = 'PtydScan'
    p.scans.MF.data.source = 'file'
    p.scans.MF.data.dfile = os.path.join(run_name, sample_name + '.ptyd')
    #p.scans.MF.data.orientation = 5 #try 0-7 #add to ptychoscan class
    #1 recreates same probe anomaly as before 
    #3 may need more iterations
    #4 recreates probe anomaly but rotated
    #5 works best!
    #6 works great!
    #7 does not work (error reaches 0 and everything is noise)
    
    p.scans.MF.illumination = u.Param()
    p.scans.MF.illumination.photons = None
    p.scans.MF.illumination.aperture = u.Param()
    p.scans.MF.illumination.aperture.form = 'circ'
    p.scans.MF.illumination.aperture.size = 150e-6
    p.scans.MF.illumination.aperture.central_stop = 45/150
    p.scans.MF.illumination.propagation = u.Param()
    p.scans.MF.illumination.propagation.focussed = 0.0532
    p.scans.MF.illumination.propagation.parallel = -0.001 #Can try negative values too
    
    p.scans.MF.coherence = u.Param()
    p.scans.MF.coherence.num_probe_modes = probe_mode_cnt
    p.scans.MF.coherence.num_object_modes = obj_mode_cnt
    
    if probe_mode_cnt > 1:
        p.scans.MF.illumination.diversity = u.Param()
        p.scans.MF.illumination.diversity.power = 0.1
        p.scans.MF.illumination.diversity.noise = [0.5, 1.0]#, np.pi, np.pi]
    
    # attach a reconstrucion engine
    p.engines = u.Param()
    
    p.engines.engine00 = u.Param()
    p.engines.engine00.name = 'RAAR_cupy_nostream'# if probe_mode_cnt == 1 else 'DM_cupy_nostream'
    p.engines.engine00.numiter = 4000
    p.engines.engine00.numiter_contiguous = 50 
    #TODO: Look into fourier support options (fraction of frame)
    #p.engines.engine00.probe_fourier_support = 0.7
    p.engines.engine00.probe_center_tol = 0
    p.frames_per_block = 100
    
    #p.engines.engine01 = u.Param()
    #p.engines.engine01.name = 'ML_cupy'
    #p.engines.engine01.numiter = 500
    #p.engines.engine01.numiter_contiguous = 50
    #p.engines.engine01.floating_intensities = True
    
    #p.engines.engine00 = u.Param()
    #p.engines.engine00.name = 'EPIE_cupy'
    #p.engines.engine00.numiter = iter_count
    #p.engines.engine00.numiter_contiguous = 500 // probe_mode_cnt
    #p.engines.engine00.probe_center_tol = 0
    #p.engines.engine00.alpha = 0.1
    #p.engines.engine00.beta = 0.9
    
    p.engines.engine00.position_refinement = u.Param()
    p.engines.engine00.position_refinement.method = "Annealing"
    p.engines.engine00.position_refinement.start = 100
    p.engines.engine00.position_refinement.stop = 4000
    p.engines.engine00.position_refinement.interval = 10
    p.engines.engine00.position_refinement.nshifts = 16
    p.engines.engine00.position_refinement.amplitude = 75.0e-9
    p.engines.engine00.position_refinement.max_shift = 100.0e-9
    p.engines.engine00.position_refinement.record = False


    if __name__ == "__main__":
        start_time = time()

        P = Ptycho(p,level=4)
        if probe_def is not None:
            sname = P.probe.S.keys()[0]
            ph = np.angle(P.probe.S[sname].data[0])
            P.probe.S[sname].data[0] = probe_def * np.exp(1j * ph)
        P.run()

        end_time = time()
        print ("Time Taken: %e s" % (end_time - start_time))

        #sname = 'SMFG00'
        #data = np.angle(P.obj.S[sname].data[0])
        ##crop to 50% of the 
        #ccrop_x = int((data.shape[0] * 0.34) / 2)
        #ccrop_y = int((data.shape[0] * 0.34) / 2)
        #data = data[ccrop_x:-ccrop_x, ccrop_y : -ccrop_y]
        #data = data - data.min()
        #data = data / data.max()
        #data, _ = fourier_upsample(data, [1, 1], factor = 4)
        #data = np.abs(data)
        #plt.imsave(os.path.join(run_name, sample_name + '.png'), data.T, cmap='gray', vmin = 0.5)

        #e,v = u.ortho(P.probe.S[sname].data)
        #fig, axes = plt.subplots(ncols=probe_mode_cnt, nrows=2,figsize=(3 * probe_mode_cnt,6))
        #for i in range(probe_mode_cnt):
        #    axes[0,i].set_title("{:.2f} %".format(e[i]*100.))
        #    ax1 = ptypy.utils.PtyAxis(axes[0,i],channel="c")
        #    ax1.set_data(v[i][200:-200,200:-200])
        #    ax2 = ptypy.utils.PtyAxis(axes[1,i],channel="a")
        #    ax2.set_data(v[i][200:-200,200:-200])
        #plt.savefig(os.path.join(run_name, "probe_modes.png"))
        #plt.close()

        exit()

    
def main():
    v = srwl_uti_parse_options(srwl_uti_ext_options(varParam), use_sys_argv=True)
    
    #v = srwpy.srwl_bl.srwl_uti_parse_options(srwpy.srwl_bl.srwl_uti_ext_options(varParam), use_sys_argv=True)
    #names = ['CCM_C1','CCM_C2','After_CCM_Slits','Slits','Slits_Before_Ap_ZP','Ap_ZP','IL_as_ZP','After_ZP_At_Sample']
    #op = set_optics(v, names, True)
    v.ws = True
    v.ws_pl = 'xy'
    v.wm = False
    
    base_cm = 0
    cm_cnt = 5
    pulses_per_pos = 1
    #radPar = {'xlamds': 1.261043e-10, 'ncar': 181, 'dgrid': 180e-6, 'zsep': 250, 'itdp': 1, 'ntail': 0} #OC18122023
    
    #run_name = 'xpp_jitter_JF'
    computeProbe = True #Compute the probe by propagating the Genesis pulse, otherwise load the probe from a file
    applyJitter = v.scan_jitter == 1 #Apply jitter to the spectrum and transverse position of the pulse
    computeScatter = v.scatter_calc == 1 #Compute the scatter by propagating the probe, else save probe and exit
    applyDetNoise = v.scan_det_noise == 1 #Apply photon noise to the detector
    computeProbeIntens = True #Compute the intensity of the probe
    #v.fdir = run_name
    run_name = 'hxn_fc_ptycho_20250814_0'
    computeOnceOnly = True #Compute the probe and scatter only once

    pulse_idx = 1
    task_cnt = int(os.environ["SLURM_NTASKS"]) if "SLURM_NTASKS" in os.environ else 1
    local_idx = int(os.environ["SLURM_LOCALID"]) if "SLURM_LOCALID" in os.environ else 0
    save_idx_base = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else 0
    gpu_cnt = 4
    gpu_idx = 1 #(local_idx % gpu_cnt) + 1

    #v.op_Sample_file_path = 'merged_s200_c400_u8.tiff' #'resolution_test_sample.png'
    sample_name = os.path.splitext(os.path.basename(v.op_Sample_file_path))[0] + "-" + str(v.scan_det_noise)
    #v.op_Sample_resolution = 6.5e-09 #m
    #v.op_Sample_areaYEnd = v.op_Sample_areaXEnd = 3840
    scanDim = (v.scan_width, v.scan_height) #(5e-06, 5e-06) #m
    scanStep = v.scan_step #0.234e-06 #m
    spiralSteps = spiral_pattern(scanDim[0], scanDim[1], scanStep, v.op_Sample_resolution)
    #spiralSteps = spiralSteps[:1]
    if computeOnceOnly:
        spiralSteps = spiralSteps[pulse_idx:pulse_idx+1]

    if save_idx_base >= len(spiralSteps): #If the save index is out of bounds, exit
        print ("save_idx_base out of bounds, save_idx_base=", save_idx_base, "len(spiralSteps)=", len(spiralSteps))
        return
    
    if save_idx_base == 0:
        os.makedirs(v.fdir, exist_ok=True)

    if v.scan_mode == 0 and save_idx_base == 0:
        #Copy current script into the output directory
        shutil.copyfile(__file__, os.path.join(v.fdir, sample_name + "_script.py"))
        print ('task_cnt=', task_cnt, 'len(spiralSteps)=', len(spiralSteps), 'run_name=', run_name, 'sample_name=', sample_name, 'computeProbe=', computeProbe, 'applyJitter=', applyJitter, 'computeScatter=', computeScatter, 'applyDetNoise=', applyDetNoise)
    

    if v.scan_mode == -1:
        #Convert result to ptyd format
        mask = np.load(os.path.join(v.fdir, 'mask.npy'))
        convert_to_ptyd(spiralSteps, v.op_DCM_C2_energy, v.op_At_Waist_Watchpoint_L, 75e-06, v.op_Sample_resolution, os.path.join(v.fdir, 'at_detector_%s'%(sample_name), 'intensity_%d.h5'), os.path.join(run_name, sample_name), info_template = os.path.join(v.fdir, 'probes_intens', 'intensity_%d.h5'), mask=mask, frame_sz=(1024, 1024), frame_center=(375, 375))#, frame_sz=(512, 512), frame_center=(512, 512))
        return
    
    if v.scan_mode == -2:
        probe_def = None
        #data, mesh,_,_ = srwl_uti_read_intens_hdf5(os.path.join(v.fdir, 'at_detector_%s'%(sample_name), 'intensity_%d.h5'%(save_idx_base)))
        #data, mesh = srwl_uti_read_intens_ascii(os.path.join(v.fdir, 'probes_intens', 'intensity_%d.dat'%(0)))
        #probe_def = np.array(data).reshape((mesh.ny, mesh.nx))
        #Run ptypy on the generated ptyd file
        ptypy_run(run_name, sample_name, iter_count=5000, probe_mode_cnt=1 if applyJitter else 1, probe_def=probe_def)
        return

    wfr = srwl_uti_read_wfr_cm_hdf5(_file_path = 'NSLS-II_HXN_Ideal_Und_Large_Coupl_cm.h5')
    for save_idx in range(save_idx_base, len(spiralSteps), task_cnt):
        random.seed(save_idx) #Ensure that the random values per probe are predictable

        #Extract one energy slice of the pulse, propagate it through the beamline, and save the result to accumulate the final image
        print('Setting-up beamline and propagating pulse: ')#, end='')
        t0 = time()
        
        #if applyJitter and pulse_per_pos_idx > 0:
        #    v.op_AngJitter_angX += random.gauss(0, 1.3e-6)
        #    v.op_AngJitter_angY += random.gauss(0, 1.3e-6)
        #    sampleXShift_rand = random.gauss(0, 5e-9) #TODO Redo with 2nm sample position jitter within each scan position
        #    sampleYShift_rand = random.gauss(0, 5e-9)

        #Save the probe data
        #os.makedirs(os.path.join(v.fdir, 'probes_info'), exist_ok=True)
        #with open(os.path.join(v.fdir, 'probes_info', 'wfr_3d_%d_info.txt'%(save_idx)), 'w') as f:
        #    f.write('%e #Spectral Jitter (eV)\n'%(0))
        #    f.write('%e, %e #Position Jitter (rad, rad)\n'%(v.op_AngJitter_angX, v.op_AngJitter_angY))
        #    f.write('%e #Pulse Energy (uJ)\n'%(0))
        finalWfrs = []
        wfr_intens_sum = None
        for cmIdx in range(base_cm, cm_cnt):
            print("Propagating CM #", cmIdx, flush=True)
            t1 = time()
            
            curWfr = copy.deepcopy(wfr[cmIdx])

            #if applyJitter:
            #    curWfr.sim_src_offset(0, v.op_AngJitter_angX, 0, v.op_AngJitter_angY, _move_mesh=True)

            #names = ['WB_Slits','WB_Slits_M1_HFM','M1_HFM','After_M1_Before_DCM','DCM_C1','DCM_C2','After_DCM_M2_VFM','M2_VFM','After_M2_Before_SSA1','SSA1']#,'After_SSA1_Before_ZP']#,'Ap_ZP','ZP','Beam_Stop','After_ZP_Before_OSA','OSA','After_OSA_At_Sample']
            #names = ['WB_Slits','WB_Slits_M1_HFM','M1_HFM','After_M1_Before_DCM','After_DCM_M2_VFM','M2_VFM','After_M2_Before_SSA1','SSA1','After_SSA1_Before_ZP','Ap_ZP','ZP','Beam_Stop','After_ZP_Before_OSA','OSA','After_OSA_At_Sample']
            names = ['S1', 'S1_HCM', 'HCM', 'HCM_Before_DCM', 'DCM_C1', 'DCM_C2', 'After_DCM_HFM', 'HFM', 'HFM_VFM', 'VFM', 'VFM_VPM', 'VPM', 'After_VPM_Before_SSA', 'SSA']#, 'After_SSA_Before_VKB']#, 'VKB', 'After_VKB_Before_HKB', 'HKB', 'After_HKB_Focus']
            op = set_optics(v, names, want_final_propagation=False)
            #op = set_optics(v, names, want_final_propagation=True)

            srwl.PropagElecField(curWfr, op, None, gpu_idx)
            exit()
            if cmIdx > base_cm:
                srwl.ResizeElecFieldMesh(curWfr, first_mesh, [0, 0])
            else:
                first_mesh = copy.deepcopy(curWfr.mesh)
            
            #names = ['After_SSA1_Before_ZP','Ap_ZP','ZP','Beam_Stop','After_ZP_Before_OSA','OSA','After_OSA_At_Sample']
            ##names = ['After_SSA1_Before_ZP','Ap_ZP','ZP','Beam_Stop','After_ZP_Before_OSA','OSA','After_OSA_At_Sample']
            #op = set_optics(v, names, want_final_propagation=False)
            ##op = set_optics(v, names, want_final_propagation=True)
            #srwl.PropagElecField(curWfr, op, None, gpu_idx)

            px_cnt = int(2560 * 0.5)
            sideb2 = (px_cnt/2) * 0.5e-9
            orig_mesh = SRWLRadMesh(curWfr.mesh.eStart, curWfr.mesh.eFin, curWfr.mesh.ne, -sideb2, sideb2, px_cnt, -sideb2, sideb2, px_cnt, curWfr.mesh.zStart)
            srwl.ResizeElecFieldMesh(curWfr, orig_mesh, [0, 0])
            
            t0 = time()
            print('Propagating probe through sample to detector: ', end='')

            v.op_Sample_shiftX = int((spiralSteps[save_idx][0]) / v.op_Sample_resolution)
            v.op_Sample_shiftY = int((spiralSteps[save_idx][1]) / v.op_Sample_resolution)

            #names = ['Sample', 'Sample_bg', 'At_Waist_Watchpoint']
            names = ['Sample']#, 'Watchpoint_Detector']
            op = set_optics(v, names, want_final_propagation=False)
            
            srwl.PropagElecField(curWfr, op, None, gpu_idx)
            finalWfrs.append(curWfr)
            
            print('done in', round(time() - t1, 3), 's')

        print('Saving detector intensity: ', end='')
        t0 = time()

        mesh = finalWfrs[0].mesh
        nx = mesh.nx
        ny = mesh.ny

        ##arI = array('f', [0]*(nx*ny))

        ##for i in range(cm_cnt - base_cm):
        ##    if i > 0:
        ##        srwl.ResizeElecFieldMesh(finalWfrs[i], mesh, [0, 0])
        ##    srwl.CalcIntFromElecField(arI, finalWfrs[i], 6, 0, 3, mesh.eStart, 0, 0, [2]) #Normal Intensity

        ##os.makedirs(os.path.join(v.fdir, 'at_detector_%s'%(sample_name)), exist_ok=True)
        ##srwl_uti_save_intens_ascii(arI, mesh, os.path.join(v.fdir, 'at_detector_%s'%(sample_name), 'intensity_%d.h5'%(save_idx)), #Intensity
        ##                   #fpIn + '_' + repr(iModeStart) + '_' + repr(iModeEnd) + '.dat',
        ##                  _arLabels=['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Intensity'],
        ##                  _arUnits=['eV', 'm', 'm', 'ph/s/.1%bw/mm^2'])

        #det_compound = JungFrauDet(1, 2, [0.01, 0.012]) #Off-Center
        #det_compound = JungFrauDet(1, 2, [0.0, 0.0]) #Centered
        det_compound = EIGERDet(1, 2, [0.01, 0.012]) #Centered
        wfr_intens, mesh, mask = det_compound.apply(finalWfrs, exposure_time=0.01 * 1.60218e-19 * 12000 * 1e-3 / pulses_per_pos if applyDetNoise else None, ret_mask=True, sum_wfrs=True, _gpu=gpu_idx, ec = 12000)

        if wfr_intens_sum is None:
            wfr_intens_sum = wfr_intens[0]
        else:
            wfr_intens_sum += wfr_intens[0]

        wfr_intens = [wfr_intens_sum]
        if save_idx == 0:
            plt.imsave(os.path.join(v.fdir, 'intens.png'), wfr_intens[0].reshape((mesh.ny, mesh.nx)), cmap='gray')
            log_intens = np.log10(wfr_intens[0].reshape((mesh.ny, mesh.nx)), where=(wfr_intens[0].reshape((mesh.ny, mesh.nx)) > 0))
            #log_intens = np.where(mask, log_intens - np.min(log_intens), log_intens)
            plt.imsave(os.path.join(v.fdir, 'intens_log.png'), log_intens, vmax = 12 if applyDetNoise else 10, vmin = 0)
            np.save(os.path.join(v.fdir, 'mask.npy'), mask)

        #Save the detector intensity
        os.makedirs(os.path.join(v.fdir, 'at_detector_%s'%(sample_name)), exist_ok=True)
        srwl_uti_save_intens_hdf5(wfr_intens[0], mesh, os.path.join(v.fdir, 'at_detector_%s'%(sample_name), 'intensity_%d.h5'%(save_idx)))
        print (np.sum(wfr_intens[0]), 'photons')
        print('done in', round(time() - t0, 3), 's')

        #os.makedirs(os.path.join(v.fdir, 'probes_info'), exist_ok=True)
        #with open(os.path.join(v.fdir, 'probes_info', 'wfr_3d_%d_info.txt'%(save_idx)), 'w') as f:
        #    f.write('%e #Spectral Jitter (eV)\n'%(eJitter_amnt))
        #    f.write('%e, %e #Position Jitter (rad, rad)\n'%(tJitter_x, tJitter_y))
        #    f.write('%e #Pulse Energy (uJ)\n'%(pulse_energy))

if __name__ == "__main__":
    main()

#epilogue()
