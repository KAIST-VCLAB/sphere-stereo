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

typedef unsigned char uchar;

inline __device__ float length(float3 a)
{
    return norm3df(a.x, a.y, a.z);
}

inline __device__ float3 operator*(float b, float3 a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator/(float3 a, float b)
{
	float den = 1.f / b;
	return make_float3(a.x * den, a.y * den, a.z * den);
}

inline __device__ float length(float2 a)
{
	return hypotf(a.x, a.y);
}

inline __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

inline __device__ float2 operator/(float2 a, float b)
{
    float den = 1.f / b;
    return make_float2(a.x * den, a.y * den);
}

inline __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

inline __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}

inline __device__ float3 uchar3Tofloat3(uchar3 a)
{
    return make_float3((float)(a.x), (float)(a.y), (float)(a.z));
}

inline __device__ uchar3 float3Touchar3(float3 a)
{
    return make_uchar3((uchar)(a.x), (uchar)(a.y), (uchar)(a.z));
}

inline __device__ float absSum(float3 a)
{
	return fabs(a.x) + fabs(a.y) + fabs(a.z);
}
