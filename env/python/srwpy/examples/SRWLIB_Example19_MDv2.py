#############################################################################
# SRWLIB Example#19: Simulating Coherent X-ray (Gaussian beam) Scattering from ensemble
# of 3D Nano-Particles modifying their positions due to Brownian motion
# Authors: H. Goel (SBU/ECE), O.C. (BNL/NSLS-II)
# v 0.03
#############################################################################

from __future__ import print_function #Python 2.7 compatibility

try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from srwl_uti_smp import *
    import srwl_uti_smp_rnd_obj3d
    from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.srwl_uti_smp import *
    from srwpy import srwl_uti_smp_rnd_obj3d
    from srwpy.uti_plot import *
#from srwlib import *
#from srwl_uti_smp import *
#import srwl_uti_smp_rnd_obj3d
#from uti_plot import * #required for plotting
import os
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()

#Choose visible GPU by rank by setting CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 2)
#Wait a random amount of time to avoid all processes trying to access the same GPU at the same time

npIsAvail = False
try:
    import numpy as np
    npIsAvail = True
except:
    print('NumPy can not be loaded. You may need to install numpy, otherwise some functionality of this example will not be available. If you are using pip, you can use the following command to install it: \npip install numpy')

time.sleep(np.random.rand() * 10)
print('SRWLIB Python Example # 19:')
print('Simulating Coherent X-ray (Gaussian beam) Scattering from ensemble of 3D particles modifying their positions due to Brownian motion')

#***********Folder and Data File Names
strDataFolderName = 'data_example_19' #Data sub-folder name
strSampleSubFolderName = 'samples' #Sub-folder name for storing Sample data
strListSampObjFileName = 'ex19_smp_obj_list_%d.dat' #List of 3D Nano-Objects / Sample file name
strSampOptPathDifOutFileName = 'ex19_smp_opt_path_dif_%d.dat' #optical path difference corresponding to selected sample
strIntInitOutFileName = 'ex19_res_int_in.dat' #initial wavefront intensity distribution output file name
strIntPropOutFileName = 'ex19_res_int_prop_%d.dat' #propagated wavefront intensity distribution output file name
strIntPropOutFileNameDet = 'realchx_pc_500frames_v3.h5' #intensity distribution regisgtered by detector output file name
strCmDataFileName = 'chx_res_9650eV_pr_dir_100k_cm.h5' #file name for loading coherent modes from file

def uti_read_wfr_cm_hdf5(_file_path, _gen0s=True): #OC11042020
    """
    Reads-in Wavefront data file (with a number of wavefronts, calculated in the same mesh vs x nad y)
    :param _file_path: string specifying path do data file to be loaded
    :param _gen0s: switch defining whether zero-electric field array(s) need to be created if some polarization component(s) are missing in the file 
    :return: list of wavefronts (objects of SRWLWfr type)
    """
    ### Load package numpy
    try:
        import numpy as np
    except:
        raise Exception('NumPy can not be loaded. You may need to install numpy. If you are using pip, you can use the following command to install it: \npip install numpy')

    ### Load package h5py
    try:
        import h5py as h5
    except:
        raise Exception('h5py can not be loaded. You may need to install h5py. If you are using pip, you can use the following command to install it: \npip install h5py')

    wfr = SRWLWfr() #auxiliary wavefront
    mesh = wfr.mesh

    hf = h5.File(_file_path, 'r')

    #Get attributes
    ats = hf.attrs
    mesh.eStart = float(ats.get('eStart'))
    mesh.eFin = float(ats.get('eFin'))
    mesh.ne = int(ats.get('ne'))
    mesh.xStart = float(ats.get('xStart'))  
    mesh.xFin = float(ats.get('xFin'))
    mesh.nx = int(ats.get('nx'))
    mesh.yStart = float(ats.get('yStart'))
    mesh.yFin = float(ats.get('yFin'))
    mesh.ny = int(ats.get('ny'))
    mesh.zStart = float(ats.get('zStart'))

    try: #OC09052021 (to walk around cases when this input is absent)
        wfr.numTypeElFld = ats.get('numTypeElFld')
        wfr.Rx = float(ats.get('Rx'))
        wfr.Ry = float(ats.get('Ry'))
        wfr.dRx = float(ats.get('dRx'))
        wfr.dRy = float(ats.get('dRy'))
        wfr.xc = float(ats.get('xc'))
        wfr.yc = float(ats.get('yc'))
        wfr.avgPhotEn = float(ats.get('avgPhotEn'))
        wfr.presCA = int(ats.get('presCA'))
        wfr.presFT = int(ats.get('presFT'))
        wfr.unitElFld = int(ats.get('unitElFld'))
        wfr.unitElFldAng = int(ats.get('unitElFldAng'))
    except:
        wfr.numTypeElFld = 'f'
        wfr.Rx = 0
        wfr.Ry = 0
        wfr.dRx = 0
        wfr.dRy = 0
        wfr.xc = 0
        wfr.yc = 0
        wfr.avgPhotEn = 0
        wfr.presCA = 0
        wfr.presFT = 0
        wfr.unitElFld = 1
        wfr.unitElFldAng = 0

    #Get All Electric Field data sets
    arEx = None
    arExH5 = hf.get('arEx')
    if(arExH5 is not None):
        arEx = np.array(arExH5)[:50] #.reshape(-1)
        #arEx = [arEx[1], arEx[0]]
        #arEx = [arEx, arEx]

    arEy = None
    arEyH5 = hf.get('arEy')
    if(arEyH5 is not None):
        arEy = np.array(arEyH5)[:50] #.reshape(-1)
        #arEy = [arEy[1], arEy[0]]
        #arEy = [arEy, arEy]

    nWfr = 0
    lenArE = 0
    if(arEx is not None):
        nWfr = len(arEx)
        lenArE = len(arEx[0])
    elif(arEy is not None):
        nWfr = len(arEy)
        lenArE = len(arEy[0])

    
    wfrs = []
    for iWfr in range(nWfr):

        arE0s = None #OC28062021
        if(_gen0s and (lenArE > 0)): arE0s = np.zeros(lenArE, dtype=np.float32) #OC28062021
        #arE0s = None if(lenArE <= 0) else np.array([0]*lenArE, 'f')
        
        wfrN = deepcopy(wfr)
        if(arEx is not None):
            wfrN.arEx = arEx[iWfr]
        else:
            wfrN.arEx = copy(arE0s)

        if(arEy is not None):
            wfrN.arEy = arEy[iWfr]
        else:
            wfrN.arEy = copy(arE0s)
        wfrs.append(wfrN)

    return wfrs

wfr_list = uti_read_wfr_cm_hdf5(os.path.join(os.getcwd(), strCmDataFileName))
print('{} coherent modes loaded'.format(len(wfr_list)))

#************Defining Samples (lists of 3D objects (spheres))
#Initial set of 3D objects
rx = 40.e-06 #Range of Horizontal position [m] within which 3D Objects constituing Sample are defined
ry = 40.e-06 #Range of Vertical position [m]
rz = 40.e-06 #Range of Longitudinal position [m]
xc = 0 #Horizontal Center position of the Sample
yc = 0 #Vertical Center position of the Sample
zc = 0 #Longitudinal Center position of the Sample
    
#Generate timesteps of Brownian motion of the 3D nano-objects (spheres) simulating particles suspended in water at room temperature
step = 100
total_step_cnt = 500
base_i = 100

step_cnt = total_step_cnt // world_size
base_i += rank * step_cnt
if rank == world_size - 1:
    step_cnt += total_step_cnt % world_size

base_path = '/media/hgoel/SSD2/BNL/LAMMPS/sio2_0.02/converted_full_0.02/smp_%d.dat'
timeStep = 0.0001 #Time step between different Sample "snapshots" / scattering patterns
timeInterv = timeStep * (step_cnt - 1) #Total time interval covered by the "snapshots"

#Sample Material Characteristics (Au at 8 keV)
matDelta = 5.953737593267228e-06#8.686e-06#4.773e-05 #Refractive Index Decrement
matAttenLen = 0.00017889018393371512#0.0001035733#2.48644e-06 #Attenuation Length [m]

#***********Detector
nxDet = 2000#2048 #Detector Number of Pixels in Horizontal direction
nyDet = 2000#2048 #Detector Number of Pixels in Vertical direction
pSize = 75e-06 #Detector Pixel Size
xrDet = nxDet*pSize
yrDet = nyDet*pSize
det = SRWLDet(_xStart = -0.5*xrDet, _xFin = 0.5*xrDet, _nx = nxDet, _yStart = -0.5*yrDet, _yFin = 0.5*yrDet, _ny = nyDet) #OC20092021
#det = SRWLDet(_xStart = -0.5*xrDet, _xFin = 0.5*xrDet, _nx = nxDet, _yStart = -0.5*yrDet, _yFin = 0.5*yrDet, _ny = nyDet)

arDetFrames = None #Array to store all detector frames data
if(npIsAvail): arDetFrames = np.zeros((step_cnt, nxDet, nyDet))

#***********Defining Drift from Sample to Detector and Propagation Parameters
distSmp_Det = 16.718
opSmp_Det = SRWLOptD(distSmp_Det)

#Wavefront Propagation Parameters:
#[0]: Auto-Resize (1) or not (0) Before propagation
#[1]: Auto-Resize (1) or not (0) After propagation
#[2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
#[3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
#[4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
#[5]: Horizontal Range modification factor at Resizing (1. means no modification)
#[6]: Horizontal Resolution modification factor at Resizing
#[7]: Vertical Range modification factor at Resizing
#[8]: Vertical Resolution modification factor at Resizing
#[9]: Type of wavefront Shift before Resizing
#[10]: New Horizontal wavefront Center position after Shift
#[11]: New Vertical wavefront Center position after Shift
#           [0][1][2] [3][4] [5] [6] [7]  [8]  [9][10][11] 
ppSmp =     [0, 0, 1., 0, 0,0.5, 80., 0.5, 80.,  0, 0, 0]
ppSmp_Det = [0, 0, 1., 3, 0, 1., 1.,  1.,  1.,  0, 0, 0]
ppFin =     [0, 0, 1., 0, 0, 1., 1.,  1.,  1.,  0, 0, 0]

#***********Wavefront Propagation / Scattering calculation for different instances of Sample created by Brownnian motion
for it in range(step_cnt):
    file = base_path % ((it + base_i) * step)
    listObjBrownian = uti_io.read_ascii_data_rows(file, '\t')
    listObjBrownian = [o for o in listObjBrownian if o[2] < 4e-06]

    print('Calculating for Brownian motion step #', it)
    print('   Setting up Transmission optical element from input Sample data ... ', end='')
    t = time.time()
    opSmp = srwl_opt_setup_transm_from_obj3d( #Defining Sample (Transmission object)
        shape_defs = listObjBrownian, #List of 3D Nano-Object params for the current step
        delta = matDelta, atten_len = matAttenLen, #3D Nano-Object Material params
        rx = rx, ry = ry, #Range of Horizontal and Vertical position [m] within which Nano-Objects constituing the Sample are defined
        nx = 7000, ny = 7000, #Numbers of points vs Horizontal and Vertical position for the Transmission
        xc = xc, yc = yc, #Horizontal and Vertical Center positions of the Sample
        extTr = 1) #Transmission outside the grid/mesh is zero (0), or the same as on boundary (1)
    print('done in', round(time.time() - t, 3), 's')

    print('   Extracting Optical Path Difference data from Sample Transmission optical element ... ', end='')
    t = time.time()
    #opPathDif = opSmp.get_data(_typ = 3, _dep = 3)

    print('done in', round(time.time() - t, 3), 's')

    print('   Saving Optical Path Difference data from Sample Transmission optical element ... ', end='')
    t = time.time()
    #srwl_uti_save_intens_ascii(
    #    opPathDif, opSmp.mesh, os.path.join(os.getcwd(), strDataFolderName, strSampOptPathDifOutFileName%(it)), 0,
    #    ['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Optical Path Difference'], _arUnits=['eV', 'm', 'm', 'm'])
    print('done in', round(time.time() - t, 3), 's')

    #Defining "Beamline" to Propagate the Wavefront through
    opBL = SRWLOptC([opSmp, opSmp_Det], 
                    [ppSmp, ppSmp_Det, ppFin])

    wfrP_list = deepcopy(wfr_list)
    cmFrames = []
    arI1 = None
    idx = 0
    for wfrP in wfrP_list:
        print('  Propagating Wavefront #', idx)
        t = time.time()
        srwl.PropagElecField(wfrP, opBL, None, [1, "KeepGPUMap", "DiscardData"])
        print('done in', round(time.time() - t, 3), 's')

        print('   Extracting, Projecting the Propagated Wavefront Intensity on Detector and Saving it to file ... ', end='')
        t = time.time()
        mesh1 = deepcopy(wfrP.mesh)
        arI1 = np.zeros(mesh1.nx*mesh1.ny, dtype=np.float32) #array('f', [0]*mesh1.nx*mesh1.ny) #"flat" array to take 2D intensity data

        gpu_params = [1]
        if (idx > 0 and idx % 25 == 0) or idx == len(wfrP_list) - 1:
            gpu_params.append("NoSync")
        srwl.CalcIntFromElecField(arI1, wfrP, 6, 0, 3, mesh1.eStart, 0, 0, None, None, gpu_params) #extracts intensity
        cmFrames.append(arI1)
        #if cmFrames is None:
        #    cmFrames = np.array(arI1)
        #else:
        #    cmFrames += np.array(arI1)

        del wfrP.arEx
        del wfrP.arEy
        print('done in', round(time.time() - t, 3), 's')
        #srwl_uti_save_intens_ascii(
        #    arI1, mesh1, os.path.join(os.getcwd(), strDataFolderName, strIntPropOutFileName%(it)), 0,
        #    ['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Spectral Fluence'], _arUnits=['eV', 'm', 'm', 'ph/s/.1%bw/mm^2'])
        idx += 1

    cmFrames = np.sum(cmFrames, axis=0)
    stkDet = det.treat_int(cmFrames, _mesh = mesh1) #"Projecting" intensity on detector (by interpolation)
    mesh1 = stkDet.mesh
    cmFrames = stkDet.arS
    del stkDet.arS
    if(arDetFrames is not None): arDetFrames[it] = np.reshape(cmFrames, (mesh1.ny, mesh1.nx)).transpose()

    #Plotting the Results (requires 3rd party graphics package)
    print('   Plotting the results (i.e. creating plots without showing them yet) ... ', end='')

    #Sample Optical Path Diff.
    #meshS = opSmp.mesh
    #plotMeshSx = [meshS.xStart, meshS.xFin, meshS.nx]
    #plotMeshSy = [meshS.yStart, meshS.yFin, meshS.ny]
    #uti_plot2d(opPathDif, plotMeshSx, plotMeshSy, ['Horizontal Position', 'Vertical Position', 'Optical Path Diff. in Sample (Time = %.3fs)' % (it*timeStep)], ['m', 'm', 'm'])
        
    #Scattered Radiation Intensity Distribution in Log Scale
    #plotMesh1x = [mesh1.xStart, mesh1.xFin, mesh1.nx]
    #plotMesh1y = [mesh1.yStart, mesh1.yFin, mesh1.ny]
    #arLogI1 = copy(arI1)
    #nTot = mesh1.ne*mesh1.nx*mesh1.ny
    #for i in range(nTot):
    #    curI = arI1[i]
    #    if(curI <= 0.): arLogI1[i] = 0 #?
    #    else: arLogI1[i] = log(curI, 10)

    #uti_plot2d1d(arLogI1, plotMesh1x, plotMesh1y, 0, 0, ['Horizontal Position', 'Vertical Position', 'Log of Intensity at Detector (Time = %.3f s)' % (it*timeStep)], ['m', 'm', ''])

    print('done')

if rank == 0:
    rcvd = [arDetFrames[i] for i in range(step_cnt)]
    for i in range(1, world_size):
        i_step_cnt = step_cnt
        if i == world_size - 1:
            i_step_cnt += total_step_cnt % world_size
        for j in range(i_step_cnt):
            rcvd.append(comm.recv(source=i))
    arDetFrames = np.stack(rcvd, axis=0)
    if(arDetFrames is not None): #Saving simulated Detector data file
        print('   Saving all Detector data to another file (that can be used in subsequent processing) ... ', end='')
        srwl_uti_save_intens_hdf5_exp(arDetFrames, mesh1, os.path.join(os.getcwd(), strDataFolderName, strIntPropOutFileNameDet), 
            _exp_type = 'XPCS', _dt = timeStep, _dist_smp = distSmp_Det, _bm_size_x = 23e-6, _bm_size_y = 23e-6)
        print('done')
    #plt.imshow(np.log10(np.mean(np.array(arDetFrames), axis=0)))
    #plt.show()
else:
    for i in range(step_cnt):
        comm.send(arDetFrames[i], dest=0)



#uti_plot_show() #Show all plots created
