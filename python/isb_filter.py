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

class ISB_Filter:
    def __init__(self, candidate_count, resolution, device):
        """
        Fast edge-preserving filter. 
        Compile CUDA functions and allocate intermediate scales
        Args:
            candidate_count: number of channels of the quantity to be filter.
                Corresponds to the number of distance candidates for cost volumes.
            resolution: size of the image (cols, rows) (guide and cost should match)
            device: CUDA-enabled GPU used for processing
        """
        # Read and compile CUDA functions
        with open('python/vec_utils.cuh', 'r') as f:
            utils_source = f.read()
        with open('python/isb_filter.cu', 'r') as f:
            cuda_source = utils_source + f.read()
        cuda_source = cuda_source.replace("CANDIDATE_COUNT", str(candidate_count))
        module = cupy.RawModule(code=cuda_source)

        self.guide_downsample_cuda = module.get_function("guideDownsample2xKernel")
        self.guide_upsample_cuda = module.get_function("guideUpsample2xKernel")

        # Allocate intermediate scales
        cols = resolution[0]
        rows = resolution[1]
        self.scale_count = int(min(math.log2(cols), math.log2(rows)) - 1)

        self.guides = []
        self.costs = []
        for scale in range(0, self.scale_count):
            self.guides.append(
                torch.zeros([math.ceil(rows/(2**scale)), math.ceil(cols/(2**scale)), 3], 
                            dtype=torch.uint8, device=device))
            
            self.costs.append(
                torch.zeros([candidate_count, math.ceil(rows/(2**scale)), math.ceil(cols/(2**scale))], device=device))

    def apply(self, guide, cost, sigma_i, sigma_s):
        """
        Apply the filter to a cost volume (or another quantity to be smoothed)
        Args:
            guide: [rows, cols, 3] Guide for edge-preserving filtering (uint8).
                using YUV or yCbCr colour spaces usually improves guidance
            cost: [candidate_count, rows, cols] Cost volume to be aggregated (float32)
            sigma_i: Edge preservation parameter. Lower values preserve edges during cost volume filtering
            sigma_s: Smoothing parameter. Higher values give more weight to coarser scales during filtering
        Returns:
            guide: [rows, cols, 3] Filtered guide
            cost: [candidate_count, rows, cols] Filtered cost volume
        """
        self.guides[0] = guide
        self.costs[0] = cost
        var_inv_s = 1 / (2 * sigma_s * sigma_s)
        var_inv_i = 1 / (2 * sigma_i * sigma_i)

        for scale in range(1, self.scale_count):
            block_size = min(256, 2**math.ceil(math.log2(self.guides[scale].shape[0] * self.guides[scale].shape[1])))
            grid_size = math.ceil(self.guides[scale].shape[0] * self.guides[scale].shape[1] / block_size)
            self.guide_downsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale - 1].data_ptr(),
                    self.costs[scale - 1].data_ptr(),
                    self.guides[scale - 1].shape[0],
                    self.guides[scale - 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(var_inv_i)
                )
            )

        for scale in range(self.scale_count - 2, -1, -1):
            distance = 2**scale - 0.5
            weight_down = (math.exp(-(distance * distance) * var_inv_s))
            weight_up = 1 - weight_down

            block_size = min(256, 2**math.ceil(
                math.log2(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1])))
            grid_size = math.ceil(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1] / block_size)

            self.guide_upsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale + 1].data_ptr(),
                    self.costs[scale + 1].data_ptr(),
                    self.guides[scale + 1].shape[0],
                    self.guides[scale + 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(weight_up),
                    cupy.float32(weight_down),
                    cupy.float32(var_inv_i)
                )
            )

        return self.costs[0], self.guides[0]