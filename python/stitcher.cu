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
struct Intrinsics
{
    float2 fl, principal;
    float xi, alpha;
};

struct Rotation
{
    float r[3][3];
};

inline __device__ float3 matMul3x3(const float r[3][3], float3 vect)
{
	return make_float3
	(
		r[0][0] * vect.x + r[0][1] * vect.y + r[0][2] * vect.z,
		r[1][0] * vect.x + r[1][1] * vect.y + r[1][2] * vect.z,
		r[2][0] * vect.x + r[2][1] * vect.y + r[2][2] * vect.z
    );
}

/**
 * Linear interpolation and type conversion in image. 
 * Does not perform out of image boundaries check.
 */
inline __device__ float3 interp(const uchar3* sampled, float2 uv, int columns = COLS)
{
	int u1, u2, v1, v2;
	u1 = __float2int_rd(uv.x);
	v1 = __float2int_rd(uv.y);

	u2 = u1 + 1;
	v2 = v1 + 1;

	float w1, w2, w3, w4;
	float u1f = (float)u1;
	float u2f = (float)u2;
	float v1f = (float)v1;
	float v2f = (float)v2;

	w1 = (u2f - uv.x) * (v2f - uv.y);
	w2 = (u2f - uv.x) * (uv.y - v1f);
	w3 = (uv.x - u1f) * (v2f - uv.y);
	w4 = (uv.x - u1f) * (uv.y - v1f);

	float3 p1, p2, p3, p4;
	p1 = uchar3Tofloat3(sampled[v1 * columns + u1]);
	p2 = uchar3Tofloat3(sampled[v2 * columns + u1]);
	p3 = uchar3Tofloat3(sampled[v1 * columns + u2]);
	p4 = uchar3Tofloat3(sampled[v2 * columns + u2]);

	return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4);
}

/**
 * Linear interpolation and type conversion in float map. 
 * Does not perform out of image boundaries check.
 */
 inline __device__ float interpF(const float* sampled, float2 uv, int columns = COLS)
{
	int u1, u2, v1, v2;
	u1 = __float2int_rd(uv.x);
	v1 = __float2int_rd(uv.y);

	u2 = u1 + 1;
	v2 = v1 + 1;

	float w1, w2, w3, w4;
	float u1f = (float)u1;
	float u2f = (float)u2;
	float v1f = (float)v1;
	float v2f = (float)v2;

	w1 = (u2f - uv.x) * (v2f - uv.y);
	w2 = (u2f - uv.x) * (uv.y - v1f);
	w3 = (uv.x - u1f) * (v2f - uv.y);
	w4 = (uv.x - u1f) * (uv.y - v1f);

	float p1, p2, p3, p4;
	p1 = (sampled[v1 * columns + u1]);
	p2 = (sampled[v2 * columns + u1]);
	p3 = (sampled[v1 * columns + u2]);
	p4 = (sampled[v2 * columns + u2]);

	return (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4);
}

/**
 * Unproject pixels to the unit sphere following the The Double Sphere Camera Model (https://arxiv.org/abs/1807.08957)
 */
 inline __device__ float3 unproject(float2 uv, Intrinsics calib)
{
    float2 m = (uv - calib.principal) / calib.fl;

    float r2 = m.x * m.x + m.y * m.y;
    float mz = (1.f - calib.alpha * calib.alpha * r2) 
        / (calib.alpha * sqrtf(1 - (2 * calib.alpha - 1) * r2) + 1 - calib.alpha);
    float mz2 = mz * mz;

    float3 point = make_float3(m.x, m.y, mz);
    point = ((mz * calib.xi + sqrtf(mz2 + (1 - calib.xi * calib.xi) * r2)) / (mz2 + r2)) * point;
    point.z -= calib.xi;
    
    return point;
}

/**
 * Project a point in space to pixel coordinates
 */
inline __device__ float2 project(float3 point, Intrinsics calib)
{
    float d1 = length(point);

    float c = calib.xi * d1 + point.z;
    float d2 = norm3df(point.x, point.y, c);
    float norm = calib.alpha * d2 + (1.f - calib.alpha) * c;

    return (calib.fl * make_float2(point.x, point.y)) / norm + calib.principal;
}

/**
 * Project a point in space to pixel coordinates and set valid to false if the 3D point 
 * is out of the double sphere model's scope (no effect on valid otherwise)
 */
inline __device__ float2 project(float3 point, Intrinsics calib, bool& valid)
{
    float d1 = length(point);

    float c = calib.xi * d1 + point.z;
    float d2 = norm3df(point.x, point.y, c);
    float norm = calib.alpha * d2 + (1.f - calib.alpha) * c;

    float w1 = calib.alpha > 0.5 ? (1 - calib.alpha) / calib.alpha : calib.alpha / (1 - calib.alpha);
    float w2 = (w1 + calib.xi) / sqrtf(2 * w1 * calib.xi + calib.xi * calib.xi + 1);
    valid &= point.z > - w2 * d1;    

    return (calib.fl * make_float2(point.x, point.y)) / norm + calib.principal;
}

/**
 * Reproject the fisheye image to the reference view point using z-buffering, 
 * hence this function should run twice.
 * Before first run, distanceOut should be initialized to a value larger than the maximum depth
 * The method leaves holes where no depth value reprojects to.
 * distanceIn: estimated distance map at the camera view point
 * distanceOut: distance map reprojected at te reference view point. 
 *   Should be initialized to a value larger than the maximum depth
 * calib: pointer to the calibration vector. Should follow the Intrinsics' structure
 * translation: pointer to the translation from the reference view point to the camera
 */
extern "C" __global__ void reprojectDistanceKernel(const float* distanceIn, float* distanceOut, 
    const Intrinsics* calib, const float3* translation)
{
    int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < COLS * ROWS)
    {
        float2 pixel = {float(indexIn % COLS), float(indexIn / COLS)}; 

        // Find the corresponding 3D point w.r.t the reference view point
        float3 pt = unproject(pixel, *calib);
        pt = distanceIn[indexIn] * pt - *translation;

        // Find the corresponding pixel
        float2 outPx = project(pt, *calib);
        
        float distance = length(pt);
        int indexOut = __float2int_rn(outPx.y) * COLS + __float2int_rn(outPx.x);

        // Set the distance using z-buffering
        if (distance < distanceOut[indexOut])
            distanceOut[indexOut] = distance;
    }
}

/**
 * Select the best neighbour pixels for inpainting depending on the occusion direction
 * inpaintDirWeights: Encoding for a two-pixels inpainting kernel.
 * calib: pointer to the calibration vector. Should follow the Intrinsics' structure
 * translation: pointer to the translation from the reference view point to the camera
 */
 extern "C" __global__ void createInpaintingWeightsKernel(uchar2* inpaintDirWeights, 
    const Intrinsics* calib, const float3* translation)
{
    int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < COLS * ROWS)
    {
        float2 pixel = {float(indexIn % COLS), float(indexIn / COLS)}; 

        // Obtain inpainting direction v_{T*} (see Section 3.3)
        float3 unit = unproject(pixel, *calib);

        float2 pxClose = project(MIN_DIST * unit - *translation, *calib);
        float2 pxFar = project(MAX_DIST * unit - *translation, *calib);
        
        float2 inpaintDir = pxFar - pxClose;
        inpaintDir = inpaintDir / length(inpaintDir);

        // Compute the two best weights and relative pixel locations
        int2 neighbours[2];
        float weights[2] = {0.f, 0.f};
        
        // Go through all neighbouring pixels
        for(int n(-1); n <= 1; n++)
        for(int m(-1); m <= 1; m++)
        {
            if(n || m)
            {
                float2 pixDir = {float(n), float(m)};
                pixDir = pixDir / length(pixDir);

                // Compute inpainting weights w_{m, n} following Section 3.3
                float weight = dot(pixDir, inpaintDir);

                // Pick the two best weights for inpainting
                if(weight > weights[1])
                {
                    weights[0] = weights[1];
                    neighbours[0] = neighbours[1];
                    weights[1] = weight;
                    neighbours[1] = make_int2(n, m);
                }
                else if(weight > weights[0])
                {
                    weights[0] = weight;
                    neighbours[0] = make_int2(n, m);

                }
            }            
        }

        // For each pixel, we store two neighbour pixels offsets with weights used for inpainting 
        // Each neighbour pixel is encoded on eight bits the inpainting table as:
        // |o|o|o|o|o|o|o|o|
        // |weight | y | x |
        // Where y and x are pixel displacements w.r.t the current pixel
        uchar valx = ((uchar)(weights[0] * 255.f) & 240) + 4 * uchar(1 + neighbours[0].y)  + uchar(1 + neighbours[0].x);
        uchar valy = ((uchar)(weights[1] * 255.f) & 240) + 4 * uchar(1 + neighbours[1].y)  + uchar(1 + neighbours[1].x);
        inpaintDirWeights[indexIn] = make_uchar2(valx, valy);
    }
}

/**
 * Apply the inpainting kernel to iteratively fill holes.
 * Uses the two neighbour pixels and their weights stored in inpaintDirWeights
 * to propagate the distance values to the current pixel.
 * A pixel is considered empty (and therefore requires inpainting)
 * when distanceMap[indexIn] >= MAX_DIST + 0.1.
 * distanceMap: Reprojected distance map with occlusion-holes
 * inpaintDirWeights: Encoding for a two-pixels inpainting kernel.
 */
 extern "C" __global__ void inpaintKernel(float* distanceMap, const uchar2* inpaintDirWeights)
{
    int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < COLS * ROWS && distanceMap[indexIn] >= MAX_DIST + 0.1)
    {
        int2 currentPixel = {indexIn % COLS, indexIn / COLS}; 
        
        if (currentPixel.x < COLS - 1 && currentPixel.y < ROWS - 1 && currentPixel.y >= 1 && currentPixel.x >= 1)
        {
            // Decode the two neighbour pixels of the inpainting table
            uchar2 dirWeights = inpaintDirWeights[indexIn];
            int2 neighbours[2] = {
                currentPixel + make_int2(dirWeights.x & 3, (dirWeights.x & 12) / 4) - 1,
                currentPixel + make_int2(dirWeights.y & 3, (dirWeights.y & 12) / 4) - 1
            };

            float weights[2] = {
                float(dirWeights.x & 240),
                float(dirWeights.y & 240),
            };
            
            // Fill the current pixel if at least one of the neighbours in the inpainting table has a meaningful value
            float distanceVal(0.f);
            float weightSum(0.f);
            for(int s(0); s < 2; s++)
            { 
                float sampledDistance = distanceMap[neighbours[s].y * COLS + neighbours[s].x];
                if(sampledDistance <= MAX_DIST + 0.1f && weights[s] > 0.f)
                {
                    distanceVal += weights[s] * sampledDistance;
                    weightSum += weights[s];
                }
            }
            if(weightSum > 0.f)
            {
                distanceMap[indexIn] = distanceVal / weightSum;
            }
        }
    }
}

#define PI 3.14159265

/**
 * Compute the sampling location and blending weight for a pixel in the panorama
 * samplingLut: [REFERENCES_COUNT, PANO_ROWS, PANO_COLS] Output pixel coordinates 
 *   to sample in the reference images to create the panorama 
 * blendingWeights: [REFERENCES_COUNT, PANO_ROWS, PANO_COLS] 
 *   Output blending weights for each of the fisheye camera used for stitching
 * masks: [REFERENCES_COUNT, ROWS, COLS] masks of the unreliable areas. 
 * calibs: Set of REFERENCES_COUNT calibration vectors. Should follow the Intrinsics' structure
 * rotations: Set of REFERENCES_COUNT rotations. Should follow the Rotation's structure
 * translations: Set of REFERENCES_COUNT translations from the reference view point to the cameras
 */
 extern "C" __global__ void createBlendingLutKernel(float2* samplingLut, float* blendingWeights, 
	float* masks, const Intrinsics* calibs, const Rotation* rotations, const float3* translations)
{
	int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < PANO_ROWS * PANO_COLS)
    {
        // Get the sphere point corresponding to the current pixel
        float2 pixel = {float(indexIn % PANO_COLS), float(indexIn / PANO_COLS)}; 
        float phi = (float(pixel.y) + 0.5f) * PI / PANO_ROWS - PI / 2.f;
        float theta = (float(pixel.x) + 0.5f) * 2.f * PI / PANO_COLS + PI;
        float3 unitPointPanorama = 
        {
            cosf(phi) * sinf(theta),
            sinf(phi),
            cosf(phi) * cosf(theta)
        };

        float blendingWeight[REFERENCES_COUNT];
        float blendingWeightSum = 0.f;
        for(int referenceIndex = 0; referenceIndex < REFERENCES_COUNT; referenceIndex++)
        {
            float3 unitInFisheye = matMul3x3(rotations[referenceIndex].r, unitPointPanorama);
            
            // Evaluate the sampling location for this camera
            bool valid = true;
            float2 uv = project(unitInFisheye, calibs[referenceIndex], valid);
            uv.x = min(max(uv.x, 0.1f), float(COLS) - 1.1f);
            uv.y = min(max(uv.y, 0.1f), float(ROWS) - 1.1f);
            samplingLut[referenceIndex * PANO_ROWS * PANO_COLS + indexIn] = uv;

            // Evaluate the sampling location displacement for a given change in distance
            blendingWeight[referenceIndex] = 1e-8;
            
            float2 pxNear = project(MIN_DIST * unitInFisheye - translations[referenceIndex], 
                calibs[referenceIndex], valid);
            pxNear.x = min(max(pxNear.x, 0.1f), float(COLS) - 1.1f);
            pxNear.y = float(referenceIndex * ROWS) + min(max(pxNear.y, 0.1f), float(ROWS) - 1.1f);
            float2 pxFar = project(MAX_DIST * unitInFisheye - translations[referenceIndex], 
                calibs[referenceIndex], valid);
            pxFar.x = min(max(pxFar.x, 0.1f), float(COLS) - 1.1f);
            pxFar.y = float(referenceIndex * ROWS) + min(max(pxFar.y, 0.1f), float(ROWS) - 1.1f);

            if(valid && interpF(masks, pxNear) > 0.99 && interpF(masks, pxFar) > 0.99)
            {

                float2 displacementVector(pxFar - pxNear);
                float displacementStrength(length(displacementVector));

                // Compute warp-aware blending weights to merge fisheye images
                blendingWeight[referenceIndex] = 
                    expf(-displacementStrength * displacementStrength / (1e-4 * ROWS * COLS));
            }

            blendingWeightSum += blendingWeight[referenceIndex];
        }

        for(int referenceIndex = 0; referenceIndex < REFERENCES_COUNT; referenceIndex++)
        {
            blendingWeights[referenceIndex * PANO_ROWS * PANO_COLS + indexIn] = 
                blendingWeight[referenceIndex] / blendingWeightSum;
        }
    }
}

/**
 * Merge the fisheye distance maps and images from the reference cameras into a complete RGB-D panorama.
 * samplingLut: [REFERENCES_COUNT, PANO_ROWS, PANO_COLS] pixel coordinates 
 *   to sample in the reference images to create the panorama 
 * blendingWeights: [REFERENCES_COUNT, PANO_ROWS, PANO_COLS] 
 *   Blending weights for each of the fisheye camera used for stitching
 * reprojectedDistanceMaps: [REFERENCES_COUNT, ROWS, COLS] Distance maps
 *   reprojected at reference view point and inpainted
 * distanceMaps: [REFERENCES_COUNT, ROWS, COLS] Original distance maps 
 *   at the cameras' locations. Used to reproject RGB images
 * stitchingImgs: [REFERENCES_COUNT, stitchingImgsRows, stitchingImgsCols]
 *   Fisheye images used for colour stitching. 
 *   They may have a higher resolution than the distance map as it has a neglectible impact on performance.
 *   (The same number of sampling is required regardless of its resolution)
 * stitchingImgsRows, stitchingImgsCols: Resolution of the colour images sampled during stitching.
 * calibs: Set of REFERENCES_COUNT calibration vectors. Should follow the Intrinsics' structure
 * rotations: Set of REFERENCES_COUNT rotations. Should follow the Rotation's structure
 * DistancePanorama: [PANO_ROWS, PANO_COLS] Output distance panorama stitched from reprojectedDistanceMaps
 * RGBPanorama: [PANO_ROWS, PANO_COLS] Output colour panorama stitched from stitchingImgs
 */
extern "C" __global__ void mergeRGBDPanoramaKernel(
    const float2* samplingLut, const float* blendingWeights, 
    const float* reprojectedDistanceMaps, const float* distanceMaps, 
    const uchar3* stitchingImgs, int stitchingImgsRows, int stitchingImgsCols, 
    const float3* translations, const Intrinsics* calibs, 
    float* DistancePanorama, uchar3* RGBPanorama)
{
	int indexIn = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexIn < PANO_ROWS * PANO_COLS)
    {
        float avgInvDistance = 0.f;
        float3 RGB = {0.f, 0.f, 0.f};
    
        for(int referenceIndex = 0; referenceIndex < REFERENCES_COUNT; referenceIndex++)
        {
            // Read blending LuTs
            float blendingWeight = blendingWeights[referenceIndex * PANO_ROWS * PANO_COLS + indexIn];
            float2 uv = samplingLut[referenceIndex * PANO_ROWS * PANO_COLS + indexIn];
            
            // Sample and reprojected distance and update the blended output distance
            float reprojectedDistance = interpF(reprojectedDistanceMaps, 
                uv + make_float2(0.f, float(referenceIndex * ROWS)));
            avgInvDistance += blendingWeight * 1.f / reprojectedDistance;
                
            
            float distanceForColorReprojection = interpF(distanceMaps, 
                uv + make_float2(0.f, float(referenceIndex * ROWS)));
            distanceForColorReprojection = min(reprojectedDistance, distanceForColorReprojection);

            // Find the corresponding 3D point w.r.t the reference view point
            float3 pt = unproject(uv, calibs[referenceIndex]);
            pt = distanceForColorReprojection * pt + translations[referenceIndex];
    
            // Find the corresponding pixel
            float2 rgbUV = project(pt, calibs[referenceIndex]);

            // Scale as the stitching colour images may have a different resolution
            float2 stitchingCalibRatio = {float(stitchingImgsCols) / COLS, float(stitchingImgsRows) / ROWS}; 
            rgbUV = rgbUV * stitchingCalibRatio;
            rgbUV.x = min(max(rgbUV.x, 0.1f), float(stitchingImgsCols) - 1.1f);
            rgbUV.y = min(max(rgbUV.y, 0.1f), float(stitchingImgsRows) - 1.1f);
            rgbUV.y += float(referenceIndex * stitchingImgsRows);

            // update the blended output colour
            RGB = RGB + blendingWeight * interp(stitchingImgs, rgbUV, stitchingImgsCols);
        }
    
        DistancePanorama[indexIn] = 1.f / avgInvDistance;
        RGBPanorama[indexIn] = float3Touchar3(RGB);    
    }
}