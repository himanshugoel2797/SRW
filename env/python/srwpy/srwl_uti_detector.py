# Implements utility methods for modeling of detectors in SRW

from __future__ import print_function #Python 2.7 compatibility
try:
    from srwlib import *
    from srwlpy import *
except:
    from .srwlib import *
    from .srwlpy import *

import matplotlib.pyplot as plt
from array import *

class SRWLDetChip:
    """Represents a detector chip"""
    def __init__(self, physical_bounds, image_position, saturation_value=None, _unitE=2):
        """
        :param pixel_size: pixel size in meters
        :param active_pixels: number of active pixels
        :param position: position of the chip in meters (x, y)
        :param padding: padding around the chip in meters (left, right, top, bottom)
        :param saturation: saturation value (ph/s or ph/s/mm^2)
        """
        
        self.saturation_value = saturation_value
        self.physical_bounds = physical_bounds
        self.image_position = image_position
        self._unitE  = _unitE
    
    def apply_noise(self, intensity, mesh, exposure_time):
        """
        Apply detector noise to the intensity
        :param intensity: intensity
        :param mesh: mesh
        :param exposure_time: exposure time in seconds
        """
        #TODO Implement detector noise modeling

        try:
            import numpy as np
        except ImportError:
            raise ImportError('NumPy can not be imported. Please install NumPy to use this function.')
        
        factor = self.pixel_size[0] * 1000 * self.pixel_size[1] * 1000
        print (np.sum(intensity) * factor, "ph/s/0.1%bw")
        #Divide by energy of a single photon at the central energy
        factor /= 1.60218e-19 * 0.5 * (mesh.eStart + mesh.eFin)
        factor *= exposure_time
        #print (factor)

        #convert from J/mm^2 to ph
        intensity *= factor

        #poisson noise
        rng = np.random.default_rng(0) # Fix the seed for reproducibility
        #rng = np.random.default_rng()
        intensity = rng.poisson(intensity)

        return intensity #return in ph
    
    def apply_saturation(self, intensity, mesh):
        """
        Apply saturation to the intensity
        :param intensity: intensity
        :param mesh: mesh
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError('NumPy can not be imported. Please install NumPy to use this function.')
        return np.minimum(intensity, self.saturation_value)

    def apply(self, intensity, mesh, detector_result, detector_px_sz, mask=None, physical_origin=None, image_origin=None, exposure_time=None):
        """
        Apply detector to the intensity
        :param intensity: intensity
        :param mask: mask
        :param mesh: mesh
        :param origin: origin of the intensity in meters (x, y)
        :param exposure_time: exposure time in seconds for modeling of detector noise
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError('NumPy can not be imported. Please install NumPy to use this function.')

        try:
            from scipy.signal import convolve2d
        except ImportError:
            raise ImportError('SciPy can not be imported. Please install SciPy to use this function.')

        if physical_origin is None:
            physical_origin = [0, 0]
        if image_origin is None:
            image_origin = [0, 0]

        self.pixel_size = detector_px_sz
        res_x = (mesh.xFin - mesh.xStart) / mesh.nx
        res_y = (mesh.yFin - mesh.yStart) / mesh.ny

        cur_pos = (physical_origin[0] + self.physical_bounds[0], physical_origin[1] + self.physical_bounds[2])

        chip_xStart = cur_pos[0] - mesh.xStart
        chip_yStart = cur_pos[1] - mesh.yStart
        chip_xFin = chip_xStart + (self.physical_bounds[1] - self.physical_bounds[0])
        chip_yFin = chip_yStart + (self.physical_bounds[3] - self.physical_bounds[2])

        #print (chip_xStart, chip_xFin, chip_yStart, chip_yFin, res_x, res_y)

        #Extract the chip area for upscaling
        chip_xStart_wfrAlgnd = round(chip_xStart / res_x)
        chip_yStart_wfrAlgnd = round(chip_yStart / res_y)
        chip_xFin_wfrAlgnd = round(chip_xFin / res_x)
        chip_yFin_wfrAlgnd = round(chip_yFin / res_y)
        chip_area = intensity[chip_yStart_wfrAlgnd:chip_yFin_wfrAlgnd, chip_xStart_wfrAlgnd:chip_xFin_wfrAlgnd]

        #Average the intensity over the chip pixel area
        kernel = np.ones((round(detector_px_sz[1] / res_y), round(detector_px_sz[0] / res_x)))
        kernel /= np.sum(kernel)

        #print (chip_area.shape)
        #print (kernel.shape)

        #Resample active area to the pixel size
        det_sz_y = round((self.physical_bounds[3] - self.physical_bounds[2]) / detector_px_sz[1])
        det_sz_x = round((self.physical_bounds[1] - self.physical_bounds[0]) / detector_px_sz[0])
        if kernel.shape[0] > 1 and kernel.shape[1] > 1:
            det_pixels = convolve2d(chip_area, kernel, mode='valid')[::kernel.shape[0], ::kernel.shape[1]]
        else:
            det_pixels = chip_area

        #Apply detector noise
        if exposure_time is not None:
            det_pixels = self.apply_noise(det_pixels, mesh, exposure_time)

        #Apply saturation
        if self.saturation_value is not None:
            det_pixels = self.apply_saturation(det_pixels)

        #Insert the sensed region into the detector output
        image_xStart = self.image_position[0] + image_origin[0]
        image_yStart = self.image_position[1] + image_origin[1]
        image_xFin = image_xStart + det_sz_x
        image_yFin = image_yStart + det_sz_y

        #print (image_xStart, image_xFin, image_yStart, image_yFin)
        if image_xStart < 0:
            image_xStart = 0
        if image_yStart < 0:
            image_yStart = 0
        if image_xFin > detector_result.shape[1]:
            image_xFin = detector_result.shape[1]
            det_pixels = det_pixels[:, 0:image_xFin - image_xStart]

        if image_yFin > detector_result.shape[0]:
            image_yFin = detector_result.shape[0]
            det_pixels = det_pixels[0:image_yFin - image_yStart, :]

        detector_result[image_yStart:image_yFin, image_xStart:image_xFin] = det_pixels
        if mask is not None:
            mask[image_yStart:image_yFin, image_xStart:image_xFin] = 1

        return detector_result, mask

class SRWLDetModule:
    """Represents a detector module consisting of one or more chips"""
    def __init__(self, physical_position, image_position):
        """
        :param position: position of the module in meters (x, y)
        :param shape: shape of the module in meters (x, y)
        """
        self.chips = []
        self.physical_bounds = [0, 0, 0, 0]

        self.physical_position = physical_position
        self.image_position = image_position

    def add_chip(self, chip):
        self.chips.append(chip)

        #Update shape of the module as minimum bounding box of all chips
        self.physical_bounds = [0, 0, 0, 0]
        for chip in self.chips:
            self.physical_bounds[0] = min(self.physical_bounds[0], chip.physical_bounds[0])
            self.physical_bounds[1] = max(self.physical_bounds[1], chip.physical_bounds[1])
            self.physical_bounds[2] = min(self.physical_bounds[2], chip.physical_bounds[2])
            self.physical_bounds[3] = max(self.physical_bounds[3], chip.physical_bounds[3])

        self.physical_bounds[0] += self.physical_position[0]
        self.physical_bounds[1] += self.physical_position[0]
        self.physical_bounds[2] += self.physical_position[1]
        self.physical_bounds[3] += self.physical_position[1]

    def apply(self, intensity, mesh, detector_result, detector_px_sz, mask=None, physical_origin=None, image_origin=None, exposure_time=None):
        """
        Apply detector to the intensity
        :param intensity: intensity
        :param mask: mask
        :param mesh: mesh
        :param exposure_time: exposure time in seconds for modeling of detector noise
        """

        if physical_origin is None:
            physical_origin = [0, 0]
        if image_origin is None:
            image_origin = [0, 0]

        cur_pos = (physical_origin[0] + self.physical_position[0], physical_origin[1] + self.physical_position[1])
        cur_image_pos = (image_origin[0] + self.image_position[0], image_origin[1] + self.image_position[1])

        #print ('Module: ', cur_pos[0] / 75e-06, cur_pos[1] / 75e-06)
        #print ('Image: ', cur_image_pos[0], cur_image_pos[1])

        for chip in self.chips:
            detector_result, mask = chip.apply(intensity, mesh, detector_result, detector_px_sz, mask=mask, physical_origin = cur_pos, image_origin = cur_image_pos, exposure_time = exposure_time)
        
        return detector_result, mask
        
class SRWLDetCompound:
    """Represents a compound detector consisting of one or more modules"""
    def __init__(self, pixel_sz, centerPosition=None, filename=None, upsample_factor=1):
        """
        :param centerPosition: transverse position of the compound detector in meters (x, y)
        :param filename: filename of the detector configuration file
        """
        self.modules = []
        self.physical_bounds = [0, 0, 0, 0]
        self.pixel_sz = pixel_sz
        self.upsample_factor = upsample_factor

        if centerPosition is None:
            self.centerPosition = (0, 0)
        elif isinstance(centerPosition, (int, float)):
            self.centerPosition = (centerPosition, centerPosition)
        elif isinstance(centerPosition, (list, tuple)):
            if len(centerPosition) == 1:
                self.centerPosition = (centerPosition[0], centerPosition[0])
            elif len(centerPosition) == 2:
                self.centerPosition = centerPosition
            else:
                raise ValueError('Invalid position specification')
            
        if filename is not None:
            self.load(filename)

    def add_module(self, module):
        self.modules.append(module)
        
        for module in self.modules:
            self.physical_bounds[0] = min(self.physical_bounds[0], module.physical_bounds[0])
            self.physical_bounds[1] = max(self.physical_bounds[1], module.physical_bounds[1])
            self.physical_bounds[2] = min(self.physical_bounds[2], module.physical_bounds[2])
            self.physical_bounds[3] = max(self.physical_bounds[3], module.physical_bounds[3])

    def load(self, filename):
        """
        Load the detector configuration from a geom file
        :param filename: filename
        """
        #TODO Implement loading of detector configuration
        return

    def save(self, filename):
        """
        Save the detector configuration to a geom file
        :param filename: filename
        """
        #TODO Implement saving of detector configuration
        return

    def apply(self, wfr, exposure_time=None, ret_mask = False, sum_wfrs=False, _gpu=None, ec=None):
        """
        Apply a mask to the wavefront
        :param intensity: intensity
        :param mesh: mesh
        :param exposure_time: exposure time in seconds
        """

        try:
            import numpy as np
        except ImportError:
            raise ImportError('NumPy can not be imported. Please install NumPy to use this function.')

        offset = [0, 0]
        offset[0] = -(self.physical_bounds[0] + self.physical_bounds[1]) / 2 + self.centerPosition[0]
        offset[1] = -(self.physical_bounds[2] + self.physical_bounds[3]) / 2 + self.centerPosition[1]

        sz = [int((self.physical_bounds[1] - self.physical_bounds[0]) / self.pixel_sz[0]), int((self.physical_bounds[3] - self.physical_bounds[2]) / self.pixel_sz[1])]

        if not isinstance(wfr, list):
            wfr = [wfr]

        wfr_mesh = wfr[0].mesh
        mesh = SRWLRadMesh(wfr_mesh.eStart, wfr_mesh.eFin, wfr_mesh.ne, self.physical_bounds[0] + offset[0], self.physical_bounds[1] + offset[0], sz[0] * self.upsample_factor, self.physical_bounds[2] + offset[1], self.physical_bounds[3] + offset[1], sz[1] * self.upsample_factor)
        det_mesh = SRWLRadMesh(wfr_mesh.eStart, wfr_mesh.eFin, wfr_mesh.ne, self.physical_bounds[0] + offset[0], self.physical_bounds[1] + offset[0], sz[0], self.physical_bounds[2] + offset[1], self.physical_bounds[3] + offset[1], sz[1])
        #print("Detector mesh: ", mesh.xStart, mesh.xFin, mesh.yStart, mesh.yFin, mesh.nx, mesh.ny)

        for i in range(len(wfr)):
            ResizeElecFieldMesh(wfr[i], mesh, [0, 0])
        
        ret_intensity = None
        ret_intensities = []

        for i in range(len(wfr)):
            intensity = array('f', [0]*mesh.nx*mesh.ny)

            if wfr[i].mesh.ne > 1:
                if ec is None:
                    ec = 0.5*(wfr[i].mesh.eStart + wfr[i].mesh.eFin)
                CalcIntFromElecField(intensity, wfr[i], 6, 7, 3, ec, 0, 0, _gpu)
                det_mesh.eStart = ec
                det_mesh.eFin = ec
                det_mesh.ne = 1
            else:
                CalcIntFromElecField(intensity, wfr[i], 6, 0, 3, mesh.eStart, 0, 0, _gpu)

            if sum_wfrs:
                if i == 0:
                    ret_intensity = np.array(intensity).reshape((mesh.ny, mesh.nx))
                else:
                    ret_intensity += np.array(intensity).reshape((mesh.ny, mesh.nx))
                print("Summing WFRs...")
            else:
                ret_intensities.append(np.array(intensity).reshape((mesh.ny, mesh.nx)))

        if sum_wfrs:
            intensity = ret_intensity.reshape((mesh.ny, mesh.nx))
            detector = np.zeros((sz[1], sz[0]))

            if ret_mask:
                mask = np.zeros((sz[1], sz[0]))
            else:
                mask = None

            for module in self.modules:
                detector, mask = module.apply(intensity, mesh, detector, self.pixel_sz, mask = mask, physical_origin = offset, exposure_time = exposure_time)

            if ret_mask:
                return [detector.reshape(-1)], det_mesh, mask
            else:
                return [detector.reshape(-1)], det_mesh, None
        else:
            intensities = []
            for intensity in ret_intensities:
                detector = np.zeros((sz[1], sz[0]))

                if ret_mask:
                    mask = np.zeros((sz[1], sz[0]))
                else:
                    mask = None

                for module in self.modules:
                    detector, mask = module.apply(intensity, mesh, detector, self.pixel_sz, mask = mask, physical_origin = offset, exposure_time = exposure_time)

                intensities.append(detector.reshape(-1))

            if ret_mask:
                return intensities, det_mesh, mask
            else:
                return intensities, det_mesh, None

class JungFrauModule(SRWLDetModule):
    """Represents a JungFrau detector module"""
    def __init__(self, physical_position, image_position):
        """
        :param position: position of the module in meters (x, y)
        :param shape: shape of the module in meters (x, y)
        """
        super(JungFrauModule, self).__init__(physical_position, image_position)

        #JungFrau has 2x4 chips, each 256x256 pixels with 75um pixel size and 2 pixel gap

        px_sz = 75e-06
        chip_px = 256
        chip_sz = px_sz * chip_px

        gap_px = 2
        gap_sz = px_sz * gap_px

        phys_pos_y = 0
        img_pos_y = 0
        for _ in range(2):
            phys_pos_x = 0
            img_pos_x = 0

            for _ in range(4):
                chip = SRWLDetChip([phys_pos_x, phys_pos_x + chip_sz, phys_pos_y, phys_pos_y + chip_sz], [img_pos_x, img_pos_y])
                self.add_chip(chip)

                phys_pos_x += chip_sz + gap_sz
                img_pos_x += chip_px + gap_px

            phys_pos_y += chip_sz + gap_sz
            img_pos_y += chip_px + gap_px

class JungFrauDet(SRWLDetCompound):
    """Represents a JungFrau detector"""
    def __init__(self, module_nx, module_ny, centerPosition=None, filename=None, upsample_factor=1, _customModuleGap=None):
        """
        :param module_nx: number of modules in x direction
        :param module_ny: number of modules in y direction
        :param centerPosition: transverse position of the compound detector in meters (x, y)
        :param filename: filename of the detector configuration file
        :param _customModuleGap: (x, y) gap (in pixels) to be used between modules if not the standard 7 pixels
        """
        super(JungFrauDet, self).__init__([75e-06, 75e-06], centerPosition, filename, upsample_factor)

        #JungFrau has 2x4 modules, each 512x512 pixels with 75um pixel size and 2 pixel gap

        px_sz = 75e-06
        img_x_gap = 7
        img_y_gap = 7
        if _customModuleGap is not None:
            img_x_gap = _customModuleGap[0]
            img_y_gap = _customModuleGap[1]
        phys_x_gap = img_x_gap * px_sz
        phys_y_gap = img_y_gap * px_sz
        

        phys_pos_y = 0
        img_pos_y = 0
        for _ in range(module_ny):
            phys_pos_x = 0
            img_pos_x = 0

            for _ in range(module_nx):
                module = JungFrauModule([phys_pos_x, phys_pos_y], [img_pos_x, img_pos_y])
                self.add_module(module)

                phys_pos_x = module.physical_bounds[1] + phys_x_gap
                img_pos_x += int((module.physical_bounds[1] - module.physical_bounds[0]) / px_sz) + img_x_gap

            phys_pos_y = module.physical_bounds[3] + phys_y_gap
            img_pos_y += int((module.physical_bounds[3] - module.physical_bounds[2]) / px_sz) + img_y_gap
            

class EIGERModule(SRWLDetModule):
    """Represents an EIGER detector module"""
    def __init__(self, physical_position, image_position):
        """
        :param position: position of the module in meters (x, y)
        :param shape: shape of the module in meters (x, y)
        """
        super(EIGERModule, self).__init__(physical_position, image_position)

        #EIGER has 2x4 chips, each 256x256 pixels with 75um pixel size and 2 px gaps which form 2 virtual pixels

        px_sz = 75e-06
        chip_px = 256
        chip_sz = px_sz * chip_px

        gap_px = 0
        gap_sz = px_sz * gap_px

        phys_pos_y = 0
        img_pos_y = 0
        for _ in range(2):
            phys_pos_x = 0
            img_pos_x = 0

            for _ in range(4):
                chip = SRWLDetChip([phys_pos_x, phys_pos_x + chip_sz, phys_pos_y, phys_pos_y + chip_sz], [img_pos_x, img_pos_y])
                self.add_chip(chip)

                phys_pos_x += chip_sz + gap_sz
                img_pos_x += chip_px + gap_px

            phys_pos_y += chip_sz + gap_sz
            img_pos_y += chip_px + gap_px

class EIGERDet(SRWLDetCompound):
    """Represents an EIGER detector"""
    def __init__(self, module_nx, module_ny, centerPosition=None, filename=None, upsample_factor=1):
        """
        :param module_nx: number of modules in x direction
        :param module_ny: number of modules in y direction
        :param centerPosition: transverse position of the compound detector in meters (x, y)
        :param filename: filename of the detector configuration file
        """
        super(EIGERDet, self).__init__([75e-06, 75e-06], centerPosition, filename, upsample_factor)

        #EIGER 1M has 1x2 modules, each 514x1030 pixels with 75um pixel size, 37 px vertical gap and 10 px horizontal gap

        px_sz = 75e-06
        img_x_gap = 7
        phys_x_gap = img_x_gap * px_sz
        img_y_gap = 37
        phys_y_gap = img_y_gap * px_sz
        

        phys_pos_y = 0
        img_pos_y = 0
        for _ in range(module_ny):
            phys_pos_x = 0
            img_pos_x = 0

            for _ in range(module_nx):
                module = EIGERModule([phys_pos_x, phys_pos_y], [img_pos_x, img_pos_y])
                self.add_module(module)

                phys_pos_x = module.physical_bounds[1] + phys_x_gap
                img_pos_x += int((module.physical_bounds[1] - module.physical_bounds[0]) / px_sz) + img_x_gap

            phys_pos_y = module.physical_bounds[3] + phys_y_gap
            img_pos_y += int((module.physical_bounds[3] - module.physical_bounds[2]) / px_sz) + img_y_gap
