/**
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
**/

#define EPS 0.01f

/**
 * Edge-preserving downsampling (see Section 3.2.1)
 * guideIn: Higher resolution guide input
 * costIn: Higher resolution cost (or any quantity to be smoothed) input
 * rowsIn, colsIn: Size of the input
 * guideOut: Two-times downscaled guide
 * costOut: Two-times downscaled cost
 * rowsOut, colsOut: Size of the downscaled image 
 *  should be as close as possible to half of rowsIn, colsIn
 */
extern "C" __global__ void guideDownsample2xKernel(
    const uchar3* guideIn, const float*  costIn, int rowsIn, int colsIn, 
    uchar3* guideOut, float* costOut, int rowsOut, int colsOut, float varInvI)
{
    int indexOut = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexOut < colsOut * rowsOut)
    {
        int x = indexOut % colsOut;
        int y = indexOut / colsOut;

        int x2 = 2*x;
        int y2 = 2*y;
        
        int2 plusUp = {x2 + 1, y2 + 1};
        plusUp.x = plusUp.x < colsIn ? plusUp.x : colsIn - 1;
        plusUp.y = plusUp.y < rowsIn ? plusUp.y : rowsIn - 1;
    
        int2 minusUp = {x2 - 1, y2 - 1}; 
        minusUp.x = minusUp.x >= 0 ? minusUp.x : 0;
        minusUp.y = minusUp.y >= 0 ? minusUp.y : 0;
    
		// Read current pixel and guide
        float3 currentGuide = uchar3Tofloat3(guideIn[y2 * colsIn + x2]);
        float3 neighboursGuide[8] = {
            uchar3Tofloat3(guideIn[minusUp.y * colsIn + minusUp.x]),
            uchar3Tofloat3(guideIn[y2 * colsIn + minusUp.x]),
            uchar3Tofloat3(guideIn[plusUp.y * colsIn +  minusUp.x]),
            uchar3Tofloat3(guideIn[plusUp.y * colsIn + x2]),
            uchar3Tofloat3(guideIn[plusUp.y * colsIn + plusUp.x]),
            uchar3Tofloat3(guideIn[y2 * colsIn + plusUp.x]),
            uchar3Tofloat3(guideIn[minusUp.y * colsIn + plusUp.x]),
            uchar3Tofloat3(guideIn[minusUp.y * colsIn + x2])
        };
    
		// Compute coeficient for bilateral downsampling
        float weights[8] = {0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f};
        float weightSum = 1.f;
        float3 inSumGuide = currentGuide;
    
        for(int i(0); i < 8; i++)
        {
            float diff = absSum(currentGuide - neighboursGuide[i]);
            weights[i] *= (__expf(-diff * diff * varInvI) + EPS);
            weightSum += weights[i];
            inSumGuide = inSumGuide + weights[i] * neighboursGuide[i];
        }
        
		// Downsample the guide
        guideOut[indexOut] = float3Touchar3(inSumGuide / weightSum);
    
		// Downsample the cost volume using the previously set coeficients
        for(int z(0); z < CANDIDATE_COUNT; z++)
        {
            float currentCost = costIn[z * rowsIn * colsIn + y2 * colsIn + x2];
            float neighboursCost[8] = {
                costIn[z * rowsIn * colsIn + minusUp.y * colsIn + minusUp.x],
                costIn[z * rowsIn * colsIn + y2 * colsIn + minusUp.x],
                costIn[z * rowsIn * colsIn + plusUp.y * colsIn + minusUp.x],
                costIn[z * rowsIn * colsIn + plusUp.y * colsIn + x2],
                costIn[z * rowsIn * colsIn + plusUp.y * colsIn + plusUp.x],
                costIn[z * rowsIn * colsIn + y2 * colsIn + plusUp.x],
                costIn[z * rowsIn * colsIn + minusUp.y * colsIn + plusUp.x],
                costIn[z * rowsIn * colsIn + minusUp.y * colsIn + x2]
            };
    
            float inSumCost(currentCost);
            for(int i(0); i < 8; i++)
            {
                inSumCost += neighboursCost[i] * weights[i];
            }
    
            costOut[z * rowsOut * colsOut + y * colsOut + x] = inSumCost / weightSum;
        }    
    }
}

/**
 * Edge-aware Upsampling (See Section 3.2.1 of the main paper and Section 1. of the supplemental document)
 * The process merges a coarse scale (guideIn/costIn) with a finer scale (guideInOut/costInOut).
 * The process preserves the edges of the higher resolution scale while smoothing.
 * guideIn: Lower resolution guide input
 * costIn: Lower resolution cost (or any quantity to be smoothed) input
 * rowsIn, colsIn: Size of the input
 * guideInOut: Input high resolution guide and output smoothed guide
 * costInOut: Input and output high resolution cost
 * rowsInOut, colsInOut: Size of the upsampled image 
 */
extern "C" __global__ void guideUpsample2xKernel(const uchar3* guideIn, const float*  costIn, int rowsIn, int colsIn, 
    uchar3* guideInOut, float* costInOut, int rowsInOut, int colsInOut, float weightUp, float weightDown, float varInvI)
{
    int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < colsIn * rowsIn)
    {
        int x = indexIn % colsIn;
        int y = indexIn / colsIn;

        int2 minusDown = {x - 1, y - 1}; 
        minusDown.x = minusDown.x >= 0 ? minusDown.x : 0;
        minusDown.y = minusDown.y >= 0 ? minusDown.y : 0;
        int2 plusDown = {x + 1, y + 1};
        plusDown.x = plusDown.x < colsIn ? plusDown.x : colsIn - 1;
        plusDown.y = plusDown.y < rowsIn ? plusDown.y : rowsIn - 1;

        int x2 = 2*x;
        int y2 = 2*y;
        int2 plusUp = {x2 + 1, y2 + 1}; 
        plusUp.x = plusUp.x < colsInOut ? plusUp.x : colsInOut - 1;
        plusUp.y = plusUp.y < rowsInOut ? plusUp.y : rowsInOut - 1;

        float weights1[9] = {1.f, 1.f, 1.f, 1.f, 8.f, 1.f, 1.f, 1.f, 1.f};
        float weights2[6] = {8.f, 8.f, 1.f, 1.f, 1.f, 1.f};
        float weights3[6] = {8.f, 8.f, 1.f, 1.f, 1.f, 1.f};
        float weights4[4] = {1.f, 1.f, 1.f, 1.f};

		// Read neighbour pixel values at coarser scale
        float3 neighboursGuideDown[3][3] = {
            {uchar3Tofloat3(guideIn[minusDown.y * colsIn + minusDown.x]),
             uchar3Tofloat3(guideIn[minusDown.y * colsIn + x]),
             uchar3Tofloat3(guideIn[minusDown.y * colsIn + plusDown.x])},
            {uchar3Tofloat3(guideIn[y * colsIn + minusDown.x]),
             uchar3Tofloat3(guideIn[y * colsIn + x]),
             uchar3Tofloat3(guideIn[y * colsIn + plusDown.x])},
            {uchar3Tofloat3(guideIn[plusDown.y * colsIn + minusDown.x]),
             uchar3Tofloat3(guideIn[plusDown.y * colsIn + x]),
             uchar3Tofloat3(guideIn[plusDown.y * colsIn + plusDown.x])},
        };

        float3 neighboursGuide1[9] = {
            neighboursGuideDown[0][0],
            neighboursGuideDown[0][1],
            neighboursGuideDown[0][2],
            neighboursGuideDown[1][0],
            neighboursGuideDown[1][1],
            neighboursGuideDown[1][2],
            neighboursGuideDown[2][0],
            neighboursGuideDown[2][1],
            neighboursGuideDown[2][2],
        };

        float3 neighboursGuide2[6] = {
            neighboursGuideDown[1][1],
            neighboursGuideDown[2][1],
            neighboursGuideDown[1][0],
            neighboursGuideDown[2][0],
            neighboursGuideDown[1][2],
            neighboursGuideDown[2][2],
        };

        float3 neighboursGuide3[6] = {
            neighboursGuideDown[1][1],
            neighboursGuideDown[1][2],
            neighboursGuideDown[0][1],
            neighboursGuideDown[0][2],
            neighboursGuideDown[2][1],
            neighboursGuideDown[2][2],
        };

        float3 neighboursGuide4[4] = {
            neighboursGuideDown[1][1],
            neighboursGuideDown[1][2],
            neighboursGuideDown[2][1],
            neighboursGuideDown[2][2],
        };

		// Read neighbour pixel values at Finer scale
        float3 currentGuideUp1 = uchar3Tofloat3(guideInOut[y2 * colsInOut + x2]);
        float3 currentGuideUp2 = uchar3Tofloat3(guideInOut[plusUp.y * colsInOut + x2]);
        float3 currentGuideUp3 = uchar3Tofloat3(guideInOut[y2 * colsInOut + plusUp.x]);
        float3 currentGuideUp4 = uchar3Tofloat3(guideInOut[plusUp.y * colsInOut + plusUp.x]);
    
        float3 inSumGuide1 = weightUp * currentGuideUp1;
        float3 inSumGuide2 = weightUp * currentGuideUp2;
        float3 inSumGuide3 = weightUp * currentGuideUp3;
        float3 inSumGuide4 = weightUp * currentGuideUp4;
        float weightSum1 = weightUp;
        float weightSum2 = weightUp;
        float weightSum3 = weightUp;
        float weightSum4 = weightUp;
    
		// Interscale bilateral upsampling coeficient computation
        for(int i(0); i < 9; i++)
        {
            float diff = absSum(neighboursGuide1[i] - currentGuideUp1);
            weights1[i] *= weightDown * (__expf(-diff * diff * varInvI) + EPS);
            inSumGuide1 = inSumGuide1 +  weights1[i] * neighboursGuide1[i];
            weightSum1 += weights1[i];
        }
        for(int i(0); i < 6; i++)
        {
            float diff = absSum(neighboursGuide2[i] - currentGuideUp2);
            weights2[i] *= weightDown * (__expf(-diff * diff * varInvI) + EPS);
            inSumGuide2 = inSumGuide2 + weights2[i] * neighboursGuide2[i];
            weightSum2 += weights2[i];
    
            diff = absSum(neighboursGuide3[i] - currentGuideUp3);
            weights3[i] *= weightDown * (__expf(-diff * diff * varInvI) + EPS);
            inSumGuide3 = inSumGuide3 + weights3[i] * neighboursGuide3[i];
            weightSum3 += weights3[i];
        }
        for(int i(0); i < 4; i++)
        {
            float diff = absSum(neighboursGuide4[i] - currentGuideUp4);
            weights4[i] *= weightDown * (__expf(-diff * diff * varInvI) + EPS);
            inSumGuide4 = inSumGuide4 + weights4[i] * neighboursGuide4[i];
            weightSum4 += weights4[i];
        }
        weightSum1 = 1.f / weightSum1;
        weightSum2 = 1.f / weightSum2; 
        weightSum3 = 1.f / weightSum3;
        weightSum4 = 1.f / weightSum4;
    
        bool setPlusY = y2 != plusUp.y;
        bool setPlusX = x2 != plusUp.x;
    
		// Guide upsampling
        guideInOut[y2 * colsInOut + x2] = float3Touchar3(weightSum1 * inSumGuide1);
        if(setPlusY)
            guideInOut[plusUp.y * colsInOut + x2] = float3Touchar3(weightSum2 * inSumGuide2);
        if(setPlusX)
            guideInOut[y2 * colsInOut + plusUp.x] = float3Touchar3(weightSum3 * inSumGuide3);
        if(setPlusY && setPlusX)
            guideInOut[plusUp.y * colsInOut + plusUp.x] = float3Touchar3(weightSum4 * inSumGuide4);
        
		// Cost upsampling
        for(int z(0); z < CANDIDATE_COUNT; z++)
        {
            float neighboursCostDown[3][3] = {
                {costIn[z * rowsIn * colsIn + minusDown.y * colsIn + minusDown.x],
                 costIn[z * rowsIn * colsIn + minusDown.y * colsIn + x],
                 costIn[z * rowsIn * colsIn + minusDown.y * colsIn + plusDown.x]},
                {costIn[z * rowsIn * colsIn + y * colsIn + minusDown.x],
                 costIn[z * rowsIn * colsIn + y * colsIn + x],
                 costIn[z * rowsIn * colsIn + y * colsIn + plusDown.x]},
                {costIn[z * rowsIn * colsIn + plusDown.y * colsIn + minusDown.x],
                 costIn[z * rowsIn * colsIn + plusDown.y * colsIn + x],
                 costIn[z * rowsIn * colsIn + plusDown.y * colsIn + plusDown.x]},
            };
    
            costInOut[z * rowsInOut * colsInOut + y2 * colsInOut + x2] = 
                (weightUp * costInOut[z * rowsInOut * colsInOut + y2 * colsInOut + x2] + 
                 weights1[0] * neighboursCostDown[0][0] + 
                 weights1[1] * neighboursCostDown[0][1] + 
                 weights1[2] * neighboursCostDown[0][2] + 
                 weights1[3] * neighboursCostDown[1][0] + 
                 weights1[4] * neighboursCostDown[1][1] + 
                 weights1[5] * neighboursCostDown[1][2] + 
                 weights1[6] * neighboursCostDown[2][0] + 
                 weights1[7] * neighboursCostDown[2][1] + 
                 weights1[8] * neighboursCostDown[2][2] 
                ) * weightSum1;
    
            
            if(setPlusY)
                costInOut[z * rowsInOut * colsInOut + plusUp.y * colsInOut + x2] = 
                    (weightUp * costInOut[z * rowsInOut * colsInOut + plusUp.y * colsInOut + x2] + 
                     weights2[0] * neighboursCostDown[1][1] + 
                     weights2[1] * neighboursCostDown[2][1] +
                     weights2[2] * neighboursCostDown[1][0] +
                     weights2[3] * neighboursCostDown[2][0] +
                     weights2[4] * neighboursCostDown[1][2] +
                     weights2[5] * neighboursCostDown[2][2]
                    ) * weightSum2;
    
            if(setPlusX)
                costInOut[z * rowsInOut * colsInOut + y2 * colsInOut + plusUp.x] = 
                    (weightUp * costInOut[z * rowsInOut * colsInOut + y2 * colsInOut + plusUp.x] + 
                     weights3[0] * neighboursCostDown[1][1] + 
                     weights3[1] * neighboursCostDown[1][2] + 
                     weights3[2] * neighboursCostDown[0][1] + 
                     weights3[3] * neighboursCostDown[0][2] + 
                     weights3[4] * neighboursCostDown[2][1] + 
                     weights3[5] * neighboursCostDown[2][2] 
                    ) * weightSum3;
    
            if(setPlusY && setPlusX)
                costInOut[z * rowsInOut * colsInOut + plusUp.y * colsInOut + plusUp.x] = 
                    (weightUp * costInOut[z * rowsInOut * colsInOut + plusUp.y * colsInOut + plusUp.x] + 
                     weights4[0] * neighboursCostDown[1][1] + 
                     weights4[1] * neighboursCostDown[1][2] +
                     weights4[2] * neighboursCostDown[2][1] + 
                     weights4[3] * neighboursCostDown[2][2]
                    ) * weightSum4;
        }    
    }
}