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
import cupy
from depth_estimation import RGBD_Estimator
from utils import parse_json_calib, read_input_images, evaluate_rgbd_panorama, save_rgbd_panorama

from pathlib import Path
import os.path 
import torch
import json
import argparse
import cv2
import numpy as np
from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="evaluation_dataset")
    parser.add_argument('--references_indices', nargs="*", type=int, default=[0, 2])
    parser.add_argument('--min_dist', type=float, default=0.55)
    parser.add_argument('--max_dist', type=float, default=100)
    parser.add_argument('--candidate_count', type=int, default=32)
    parser.add_argument('--sigma_i', type=float, default=10)
    parser.add_argument('--sigma_s', type=float, default=25)
    parser.add_argument('--matching_resolution', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('--rgb_to_stitch_resolution', nargs=2, type=int, default=[1216, 1216])
    parser.add_argument('--panorama_resolution', nargs=2, type=int, default=[2048, 1024])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--saving', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--bad_px_ratio_thresholds', type=float, default=[0.1, 0.4])
    args = parser.parse_args()

    f = open(os.path.join(args.dataset_path, "calibration.json"))
    raw_calibration = json.load(f)['value0']
    calibrations = parse_json_calib(raw_calibration, args.matching_resolution, args.device)

    # Reference viewpoint for the estimated RGB-D panorama is the center of the references
    reprojection_viewpoint = torch.zeros([3], device=args.device)
    for references_index in args.references_indices:
        reprojection_viewpoint += calibrations[references_index].rt[:3, 3] / len(args.references_indices)

    # Read masks
    masks = []
    for cam_index in range(len(calibrations)):
        if os.path.isfile(os.path.join(args.dataset_path, "cam" + str(cam_index)) + "/" + "mask.png"):
            mask = cv2.imread(os.path.join(args.dataset_path, "cam" + str(cam_index)) + "/" + "mask.png", 
                              cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, tuple(args.matching_resolution), cv2.INTER_AREA)
            masks.append(torch.tensor(mask, device=args.device, dtype=torch.float32).unsqueeze(0)/255)
        else:
            masks.append(torch.ones(args.matching_resolution, device=args.device).unsqueeze(0))

    # Initialize distance estimator and stitcher
    rgbd_estimator = RGBD_Estimator(calibrations, args.min_dist, args.max_dist, args.candidate_count, 
                                    args.references_indices, reprojection_viewpoint, masks, 
                                    args.matching_resolution, args.rgb_to_stitch_resolution, args.panorama_resolution, 
                                    args.sigma_i, args.sigma_s, args.device)


    filenames = os.listdir(os.path.join(args.dataset_path, "cam0/"))
    try:
        filenames.remove("mask.png")
    except ValueError:
        pass  # mask is not mandatory

    all_fisheye_images = Parallel(n_jobs=-1, backend="threading")(
        delayed(read_input_images)(
            filename, args.dataset_path, args.matching_resolution, args.rgb_to_stitch_resolution, 
            calibrations, args.references_indices) 
        for filename in filenames)

    rgbd_panoramas = {}
    for frame_index, filename in enumerate(filenames):
        fisheye_images = all_fisheye_images[frame_index]["images_to_match"]
        reference_fisheye_images = all_fisheye_images[frame_index]["images_to_stitch"]
        valid_frame = all_fisheye_images[frame_index]["is_valid"]

        if valid_frame:
            fisheye_images = [torch.tensor(fisheye_image, device=args.device) for fisheye_image in fisheye_images]
            reference_fisheye_images = [torch.tensor(reference_fisheye_image, device=args.device) 
                                        for reference_fisheye_image in reference_fisheye_images]
            rgb, distance = rgbd_estimator.estimate_RGBD_panorama(fisheye_images, reference_fisheye_images)
            
            rgbd_panoramas[filename] = {"rgb": rgb.cpu().numpy(), "inv_distance": 1 / distance.cpu().numpy()}

            if args.visualize:
                # Map inverse distance to [0, 255] and display
                distance_map = 1 / distance.cpu().numpy()
                distance_map = ((rgbd_panoramas[filename]["inv_distance"] - 1 / args.max_dist) 
                                / (1 / args.min_dist - 1 / args.max_dist))
                distance_map = np.clip(255 * distance_map, 0, 255).astype(np.uint8)
                distance_map = cv2.applyColorMap(distance_map, cv2.COLORMAP_MAGMA)
                cv2.imshow("distance_map", distance_map)
                cv2.imshow("rgb", rgbd_panoramas[filename]["rgb"])
                cv2.waitKey()

    if args.saving:
        Path(os.path.join(args.dataset_path, "output")).mkdir(parents=True, exist_ok=True)
        Parallel(n_jobs=-1, backend="threading")(
            delayed(save_rgbd_panorama)(rgbd_panoramas, filename, args.dataset_path) 
            for filename in filenames)


    if args.evaluate:
        evaluations = Parallel(n_jobs=-1, backend="threading")(
            delayed(evaluate_rgbd_panorama)(rgbd_panoramas, filename, args.dataset_path, 
                                            args.bad_px_ratio_thresholds, args.panorama_resolution) 
            for filename in filenames)

        # Average the evaluation metrics
        psnr = 0
        ssim = 0
        rmse = 0
        mae = 0
        bad_px_ratios = [0] * len(args.bad_px_ratio_thresholds)
        evaluation_count = 0

        for evaluation in evaluations:
            if evaluation is not None:
                psnr += evaluation["psnr"]
                ssim += evaluation["ssim"]
                rmse += evaluation["rmse"]
                mae += evaluation["mae"]
                bad_px_ratios = [bad_px_ratio + current_bad_px_ratio 
                    for  bad_px_ratio, current_bad_px_ratio in zip(bad_px_ratios, evaluation["bad_px_ratios"])]
                evaluation_count += 1

        if evaluation_count > 0:
            print("PSNR = ", psnr / evaluation_count)
            print("SSIM = ", ssim / evaluation_count)
            for bad_px_ratio, bad_px_ratio_threshold in zip(bad_px_ratios, args.bad_px_ratio_thresholds):
                print(">", bad_px_ratio_threshold, " = ", bad_px_ratio / evaluation_count)
            print("MAE = ", mae / evaluation_count)
            print("RMSE = ", rmse / evaluation_count)