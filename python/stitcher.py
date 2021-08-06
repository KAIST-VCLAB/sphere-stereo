"""
=======================================================================
General Information
-------------------
This is a GPU-based python implementation of the following paper:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
Visit our project http://vclab.kaist.ac.kr/cvpr2021p1/ for more details.

Please cite this paper if you use this code in an academic publication.
Bibtex: 
@InProceedings{Meuleman_2021_CVPR,
    author = {Andreas Meuleman and Hyeonjoong Jang and Daniel S. Jeon and Min H. Kim},
    title = {Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images},
    booktitle = {CVPR},
    month = {June},
    year = {2021}
}
==========================================================================
License Information
-------------------
CC BY-NC-SA 3.0
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, “Non-Commercial Purpose” means for the purpose of education or research in a non-commercial organisation only. “Non-Commercial Purpose” excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
Please refer to license.txt for more details.
=======================================================================
"""
import torch 
import cupy
import math

def vectorize_calibration(calibration, device):
    """
    Convert the intrinsics into a continuous float vector that follows the Intrinsics' structure
    (See stitcher.cu for Intrinsics' definition)
    Scale the focal length and the principal point using the matching scale.
    """
    calibration_vector = torch.zeros([6], device=device)
    calibration_vector[0:2] = calibration.fl * calibration.matching_scale
    calibration_vector[2:4] = calibration.principal * calibration.matching_scale
    calibration_vector[4] = calibration.xi
    calibration_vector[5] = calibration.alpha
    return calibration_vector

class Stitcher:
    def __init__(self, calibrations, reprojection_viewpoint, masks, min_dist, max_dist, 
                 matching_resolution, rgb_to_stitch_resolution, panorama_resolution, 
                 device, smoothing_radius = 15, inpainting_iterations = 32):
        """
        Stitcher to create RGB-D panoramas from RGB-D fisheye images.
        Compile CUDA functions, allocate intermediate arrays and compute tables
        Args:
            calibrations: [number of references] calibration parameters following the double sphere model for each camera
            reprojection_viewpoint: [3] Reference viewpoint where the RGB-D panorama will be created
            masks: [number of references][matching_rows, matching_cols] Mask of the valid area in the captured fisheye image.
            min_dist, max_dist: minimum and maximum distance expected in the distance maps
            matching_resolution: Resolution (cols, rows) used for matching. May be lower than original to save computation
            rgb_to_stitch_resolution: Resolution (cols, rows) of the colour images sampled during stitching. 
                May have a higher resolution as it has a negligible impact on performance.
                (The same number of sampling is required regardless of its resolution)
            panorama_resolution: Resolution (cols, rows) of the output RGB-D panoramas
            device: CUDA-enabled GPU used for processing
            smoothing_radius: Blending weights may have sharp discontinuities. To make sure images are blended smoothly, 
                we apply a simple box filter.
            inpainting_iterations: Number of times the inpainting is applied. Each inpainting pass fills an invalid pixel
                that is next to a valid pixel. This value should be larger than the widest occluded regions. 
        """
        self.max_dist = max_dist
        self.inpainting_iterations = inpainting_iterations
        matching_cols = matching_resolution[0]
        matching_rows = matching_resolution[1]
        self.block_size = 256
        self.fisheye_grid_size = math.ceil((matching_cols * matching_rows) / self.block_size)
        self.panorama_grid_size = math.ceil((panorama_resolution[0] * panorama_resolution[1]) / self.block_size)
        
        reprojection_viewpoint = torch.cat([reprojection_viewpoint, torch.ones([1], device=device)])

        # Read and compile CUDA functions
        with open('python/vec_utils.cuh', 'r') as f:
            utils_source = f.read()
        with open('python/stitcher.cu', 'r') as f:
            cuda_source = utils_source + f.read()

        cuda_source = cuda_source.replace("PANO_COLS", str(panorama_resolution[0]))
        cuda_source = cuda_source.replace("PANO_ROWS", str(panorama_resolution[1]))
        cuda_source = cuda_source.replace("COLS", str(matching_cols))
        cuda_source = cuda_source.replace("ROWS", str(matching_rows))
        cuda_source = cuda_source.replace("REFERENCES_COUNT", str(len(calibrations)))
        cuda_source = cuda_source.replace("MIN_DIST", str(min_dist))
        cuda_source = cuda_source.replace("MAX_DIST", str(max_dist))
        module = cupy.RawModule(code=cuda_source)
        
        self.reproject_distance_cuda = module.get_function('reprojectDistanceKernel')
        self.inpaint_cuda = module.get_function('inpaintKernel')
        self.merge_rgbd_panorama_cuda = module.get_function('mergeRGBDPanoramaKernel')
        create_inpainting_weights_cuda = module.get_function('createInpaintingWeightsKernel')
        create_blending_luts_cuda = module.get_function('createBlendingLutKernel')

        # Allocate tables and intermediate arrays
        self.reprojected_distances = torch.zeros([len(calibrations), matching_rows, matching_cols], device=device)
        self.distances_stacked = torch.zeros([len(calibrations), matching_rows, matching_cols], device=device)
        self.images_to_stitch = torch.zeros(
            [len(calibrations), rgb_to_stitch_resolution[1], rgb_to_stitch_resolution[0], 3], 
            dtype=torch.uint8, device=device)
        
        self.reprojected_distances_list = [self.reprojected_distances[i] for i in range(len(calibrations))]
        self.distances_list = [self.distances_stacked[i] for i in range(len(calibrations))]
        self.images_to_stitch_list = [self.images_to_stitch[i] for i in range(len(calibrations))]
        self.inpainting_weights_list = [
            torch.zeros([matching_rows, matching_cols, 2], dtype=torch.uint8, device=device)
            for _ in range(len(calibrations))]
        
        self.blending_sampling = torch.zeros(
            [len(calibrations), panorama_resolution[1], panorama_resolution[0], 2], device=device)
        self.blending_weights = torch.zeros(
            [len(calibrations), panorama_resolution[1], panorama_resolution[0]], device=device)

        self.RGB_panorama = torch.zeros(
            [panorama_resolution[1], panorama_resolution[0], 3], dtype=torch.uint8, device=device)
        self.distance_panorama = torch.zeros([panorama_resolution[1], panorama_resolution[0]], device=device)

        # Create tables for inpainting
        self.translations_list = []
        self.calibration_vectors_list = []
        for calibration, inpainting_weight in zip(calibrations, self.inpainting_weights_list):
            calibration_vector = vectorize_calibration(calibration, device)
            translation = torch.matmul(torch.inverse(calibration.rt), reprojection_viewpoint)[:3]
            self.translations_list.append(translation)
            self.calibration_vectors_list.append(calibration_vector)

            create_inpainting_weights_cuda(
                block=(self.block_size,),
                grid=(self.fisheye_grid_size,),
                args=(inpainting_weight.data_ptr(), 
                calibration_vector.data_ptr(),
                translation.data_ptr()))

        # Create tables for merging fisheye images into panoramas
        rotations = [torch.inverse(calibration.rt[:3, :3]) for calibration in calibrations]
        rotations = torch.cat(rotations, dim=0).contiguous()
        masks = torch.cat(masks, dim=0).contiguous()
        masks = torch.nn.functional.pad(masks.unsqueeze(1), (smoothing_radius,)*4, mode='constant', value=1)
        conv_kernel = torch.ones([1, 1, 2 * smoothing_radius + 1, 2 * smoothing_radius + 1], device=device)
        conv_kernel /= torch.sum(conv_kernel)
        masks = torch.nn.functional.conv2d(masks, conv_kernel)
        self.calibration_vectors = torch.cat(self.calibration_vectors_list, dim=0).contiguous()
        self.translations = torch.cat(self.translations_list, dim=0).contiguous()
        create_blending_luts_cuda(
            block=(self.block_size,),
            grid=(self.panorama_grid_size,),
            args=(self.blending_sampling.data_ptr(), 
            self.blending_weights.data_ptr(),
            masks.data_ptr(),
            self.calibration_vectors.data_ptr(),
            rotations.data_ptr(),
            self.translations.data_ptr()))

        # Smooth to avoid strong seems
        self.blending_weights = torch.nn.functional.conv2d(self.blending_weights.unsqueeze(1), 
                                                           conv_kernel, padding=smoothing_radius)
        self.blending_weights /= torch.sum(self.blending_weights, dim=0, keepdim=True)

    def stitch(self, images, distance_maps):
        """
        Stitch a set of colour images and distance maps into an RGB-D panorama
        Args:
            images: [number of references][rgb_to_stitch_rows, rgb_to_stitch_cols, 3], uint8 
                Colour fisheye images to be stitched. Should match rgb_to_stitch_resolution given in __init__
            distance_maps: [number of references][matching_rows, matching_cols], float32
                Fisheye distance maps to be stitched. Should match matching_resolution given in __init__
        Returns:
            rgb: [rows, cols, 3] Colour panorama as uint8
            distance: [rows, cols] Distance panorama as float32
        """
        for calibration_vector, translation, image, distance_map, distance_stack, \
                    reprojected_distance, image_to_stitch, inpainting_weight \
                in zip(self.calibration_vectors_list, self.translations_list, images, distance_maps, self.distances_list,
                    self.reprojected_distances_list, self.images_to_stitch_list, self.inpainting_weights_list):

            # Reproject the distance map to a reference view point
            reprojected_distance.fill_(1e8)
            for _ in range(2):
                self.reproject_distance_cuda(
                    block=(self.block_size,),
                    grid=(self.fisheye_grid_size,),
                    args=(distance_map.data_ptr(), 
                          reprojected_distance.data_ptr(), 
                          calibration_vector.data_ptr(),
                          translation.data_ptr()))

            # Fill-in the holes in the distance map in a background-to-foreground manner
            for _ in range(self.inpainting_iterations):
                self.inpaint_cuda(
                    block=(self.block_size,),
                    grid=(self.fisheye_grid_size,),
                    args=(reprojected_distance.data_ptr(), 
                          inpainting_weight.data_ptr()))
                
            image_to_stitch.copy_(image)
            distance_stack.copy_(distance_map)

        self.merge_rgbd_panorama_cuda(
            block=(self.block_size,),
            grid=(self.panorama_grid_size,),
            args=(self.blending_sampling.data_ptr(),
            self.blending_weights.data_ptr(),
            self.reprojected_distances.data_ptr(),
            self.distances_stacked.data_ptr(),
            self.images_to_stitch.data_ptr(),
            self.images_to_stitch.shape[1],
            self.images_to_stitch.shape[2],
            self.translations.data_ptr(),
            self.calibration_vectors.data_ptr(),
            self.distance_panorama.data_ptr(),
            self.RGB_panorama.data_ptr()))

        return self.RGB_panorama, self.distance_panorama
