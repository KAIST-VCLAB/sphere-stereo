  
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

from isb_filter import ISB_Filter
from stitcher import Stitcher
from utils import project, unproject, rgb2yCbCr
import torch

class RGBD_Estimator:
    def __init__(self, calibrations, min_dist, max_dist, candidate_count, references_indices, reprojection_viewpoint, 
                 masks, matching_resolution, rgb_to_stitch_resolution, panorama_resolution, sigma_i, sigma_s, device):
        """
        Prepare RGB-D estimation from fisheye images. 
        Perform camera selection for adaptive matching, initialize filters and stitcher 
        Args:
            calibrations: [number of cameras] calibration parameters following the double sphere model for each camera
            min_dist, max_dist: minimum and maximum distance for the sphere sweep volume computation
            candidate_count: Number of distance candidates between min_dist and max_dist (included)
            references_indices: [number of references] Indices of the cameras where distance estimation is performed before stitching 
            reprojection_viewpoint: [3] Reference viewpoint where the RGB-D panorama will be created
            masks: [number of cameras][matching_rows, matching_cols] Mask of the valid area in the captured fisheye image.
                one represents a reliable area while zero-pixels are ignored for matching.
                Typically, the camera body and the outskirt of a fisheye image are inexploitable for stereo
            matching_resolution: Resolution (cols, rows) used for matching. May be lower than original to save computation
            rgb_to_stitch_resolution: Resolution (cols, rows) of the colour images sampled during stitching. 
                May have a higher resolution as it has a negligible impact on performance.
            panorama_resolution: Resolution (cols, rows) of the output RGB-D panoramas
            sigma_i: Edge preservation parameter. Lower values preserve edges during cost volume filtering
            sigma_s: Smoothing parameter. Higher values give more weight to coarser scales during filtering
            device: CUDA-enabled GPU used for processing
        """
        self.calibrations = calibrations
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.candidate_count = candidate_count
        self.references_indices = references_indices
        self.reprojection_viewpoint = reprojection_viewpoint
        self.matching_resolution = matching_resolution
        self.device = device
        self.sigma_i = sigma_i 
        self.sigma_s = sigma_s

        self.cost_filter = ISB_Filter(candidate_count, matching_resolution, device)
        self.distance_filter = ISB_Filter(1, matching_resolution, device)

        calibrations_for_stitch = [calibrations[reference_index] for reference_index in references_indices]
        masks_for_stitching = [masks[reference_index] for reference_index in references_indices]
        self.fishey_stitcher = Stitcher(calibrations_for_stitch, reprojection_viewpoint, 
                                        masks_for_stitching, min_dist, max_dist, 
                                        matching_resolution, rgb_to_stitch_resolution, panorama_resolution, device)
        
        self.select_camera(masks)

    def select_camera(self, masks):
        """
        Select the cameras for adaptive matching (see Section 3.1)
        """
        self.selected_cameras = []
        for reference_index in self.references_indices:
            reference_calibration = self.calibrations[reference_index]
            selected_camera = -torch.ones(self.matching_resolution[::-1], dtype=int, device=self.device).unsqueeze(0)
            max_displacement = torch.ones(self.matching_resolution[::-1], device=self.device).unsqueeze(0)

            u, v = torch.meshgrid([torch.arange(0, self.matching_resolution[1], device=self.device), 
                torch.arange(0, self.matching_resolution[0], device=self.device)])
            pt_unit, reference_valid = unproject(torch.stack([v, u], dim=-1).unsqueeze(0), 
                                                 reference_calibration)

            # Go through all the matched cameras and select the best one per pixel
            for cam_index, (calibration, mask) in enumerate(zip(self.calibrations, masks)):
                pt_near = pt_unit * self.min_dist
                pt_far = pt_unit * self.max_dist

                # points in the matched camera's point of view
                rt = torch.matmul(torch.inverse(calibration.rt), reference_calibration.rt)
                pt_near = torch.matmul(torch.cat([pt_near, torch.ones_like(pt_near[..., :1])], dim=-1), rt.T)
                pt_far = torch.matmul(torch.cat([pt_far, torch.ones_like(pt_near[..., :1])], dim=-1), rt.T)
                pt_near = pt_near[..., :3] / torch.norm(pt_near[..., :3], dim=-1, keepdim=True)
                pt_far = pt_far[..., :3] / torch.norm(pt_far[..., :3], dim=-1, keepdim=True)

                uv_near, valid_near = project(pt_near, calibration)
                uv_far, valid_far = project(pt_far, calibration)  

                # Evaluate the displacement from a given change in distance
                displacement = torch.norm(uv_near - uv_far, dim=-1)
  
                # Check the validity mask of the reprojected pixels
                uv_near = ((uv_near + 0.5) / torch.tensor([self.matching_resolution[0], 
                                                   self.matching_resolution[1]], device=self.device)) * 2 - 1
                uv_far = ((uv_far + 0.5) / torch.tensor([self.matching_resolution[0], 
                                                 self.matching_resolution[1]], device=self.device)) * 2 - 1

                mask_near = torch.nn.functional.grid_sample(mask.unsqueeze(0), uv_near, align_corners=False)[0]
                mask_far = torch.nn.functional.grid_sample(mask.unsqueeze(0), uv_far, align_corners=False)[0]
                
                # Update the selected best camera
                current_best = ((displacement > max_displacement)
                                * reference_valid
                                * valid_near * valid_far
                                * (masks[reference_index] >= 0.9) * (mask_near >= 0.9) * (mask_far >= 0.9))

                max_displacement[current_best] = displacement[current_best]

                selected_camera[current_best] = cam_index
    
            self.selected_cameras.append(selected_camera)

    def estimate_fisheye_distance(self, reference_image, guide, reference_calibration, selected_camera, images):
        """
        Estimate distance on a fisheye image using the images from the other cameras
        """
        u, v = torch.meshgrid([torch.arange(0, self.matching_resolution[1], device=self.device), 
                               torch.arange(0, self.matching_resolution[0], device=self.device)])
        pt_unit, _ = unproject(torch.stack([v, u], dim=-1).unsqueeze(0), reference_calibration)
        
        distance_candidates = 1 / torch.linspace(1 / self.min_dist, 1 / self.max_dist, 
                                               self.candidate_count, device=self.device)
        point_volume = (distance_candidates.view(self.candidate_count, 1, 1, 1) * 
                        pt_unit.view(1, self.matching_resolution[1], self.matching_resolution[0], 3))
        
        sweeping_volume = torch.zeros(
            [1, 3, self.candidate_count, self.matching_resolution[1], self.matching_resolution[0]], 
            device=self.device)

        # Sweeping volume computation, with a different camera for each pixel following adaptive spherical matching 
        for cam_index, calibration in enumerate(self.calibrations):
            rt = torch.matmul(torch.inverse(calibration.rt), reference_calibration.rt)
            point_volume_in_cam = torch.matmul(torch.cat([point_volume, torch.ones_like(point_volume[..., :1])], dim=-1), 
                                               rt.T)
            uv, _ = project(point_volume_in_cam[..., :3], calibration)
            uv = ((uv + 0.5) / torch.tensor([self.matching_resolution[0], 
                                     self.matching_resolution[1]], device=self.device)) * 2 - 1
            uv = uv.unsqueeze(0)
            uv = torch.cat([uv, torch.zeros_like(uv[..., :1])], dim=-1)

            image = images[cam_index]
            sweeping_volume_for_cam = torch.nn.functional.grid_sample(image, uv, align_corners=False)
            
            selected_mask = selected_camera==cam_index
            selected_mask = selected_mask.repeat(1, 3, self.candidate_count, 1, 1)
            sweeping_volume[selected_mask] = sweeping_volume_for_cam[selected_mask]

        # Raw cost computation from difference between the sweeping volume and the image
        cost_volume = torch.sum(torch.abs(sweeping_volume - reference_image), dim=1).squeeze(0)
        # Cost volume filtering
        cost_volume = torch.clamp(cost_volume, max = 500)

        cost_volume, _ = self.cost_filter.apply(guide.clone(), cost_volume.clone(), self.sigma_i, self.sigma_s)

        # Distance selection
        min_cost, selected_index_map = torch.min(cost_volume, dim=0, keepdim=True)
        max_cost, _ = torch.max(cost_volume, dim=0, keepdim=True)

        # Quadratic fitting for sub-candidate accuracy
        left_cost = torch.gather(cost_volume, 0, 
                                 torch.clamp(selected_index_map - 1, min=0, max=self.candidate_count - 1))
        right_cost = torch.gather(cost_volume, 0, 
                                  torch.clamp(selected_index_map + 1, min=0, max=self.candidate_count - 1))
        variation = 0.5 * (left_cost - right_cost) / ((left_cost + right_cost) - 2. * min_cost + 1e-8)
        variation = torch.clamp(variation, min=-0.5, max=0.5)
        variation[selected_index_map == self.candidate_count - 1] = 0
        variation[selected_index_map == 0] = 0
        selected_index_map = selected_index_map.float() + variation
        selected_index_map[max_cost == min_cost] = self.candidate_count - 1

        # Index to distance conversion
        distance_map = distance_candidates[0] / ((distance_candidates[0] / distance_candidates[-1] - 1) 
                                                 * selected_index_map / (self.candidate_count - 1) + 1)
        distance_map[torch.abs(max_cost - min_cost) < 1e-8] = distance_candidates[-1]

        # Distance map post filtering, with higher edge preservation.
        filtered_distance, _ = self.distance_filter.apply(guide.clone(), distance_map.clone(), 
                                                          self.sigma_i/2, self.sigma_s/2)
        return filtered_distance[0]

    def estimate_RGBD_panorama(self, images_to_match, images_to_stitch):
        """
        Estimate depth on the reference fisheye images (specified when instantiating)
        Then stitch the fisheye images to produce a complete RGB-D panorama
        Args:
            images_to_match: [number of cameras][matching_rows, matching_cols, 3] Set of fisheye images for distance estimation.
                Their resolutions may be lower than original to save computation. Should be float32 with [0, 255] range
            images_to_stitch: [number of references][rgb_stitching_rows, rgb_stitching_cols, 3] Fisheye images used for colour stitching. 
                They may have a higher resolution as it has a negligible impact on performance. 
                Should be float32 with [0, 255] range
        Returns:
            rgb: [rows, cols, 3] colour panorama as uint8
            distance: [rows, cols] estimated distance panorama as float32
        """
        
        
        # Evaluate distance for each of the reference fisheye images
        images_to_match_permuted = [image.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
                                    for image in images_to_match]
        
        distance_maps = []
        for reference_index, selected_camera in zip(self.references_indices, self.selected_cameras):
            guide = rgb2yCbCr(images_to_match[reference_index]).type(torch.uint8)
            distance_maps.append(
                self.estimate_fisheye_distance(
                    images_to_match_permuted[reference_index], 
                    guide,
                    self.calibrations[reference_index], 
                    selected_camera, 
                    images_to_match_permuted, 
                )
            )

        # Stitch in a disparity aware manner to create complete panoramas
        images_to_stitch = [reference_image.type(torch.uint8)
                            for reference_image in images_to_stitch]
        rgb, distance = self.fishey_stitcher.stitch(images_to_stitch, distance_maps)

        return rgb, distance