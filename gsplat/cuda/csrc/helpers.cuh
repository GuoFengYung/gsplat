#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6

__device__ const float near_n = 0.2;
__device__ const float far_n = 100.0;
__device__ const float FilterInvSquare = 2.0f;

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 cross(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max,
    const int block_size
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)block_size, pix_center.y / (float)block_size
    };
    float2 tile_radius = {
        pix_radius / (float)block_size, pix_radius / (float)block_size
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y / 2.f, v_conic.y / 2.f, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

inline __device__ void cov2d_to_compensation_vjp(
    const float compensation, const float3 &conic, const float v_compensation, float3 &v_cov2d
) {
    // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
    // conic = inverse(cov2d)
    // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
    // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
    float inv_det = conic.x * conic.z - conic.y * conic.y;
    float one_minus_sqr_comp = 1 - compensation * compensation;
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    v_cov2d.x += v_sqr_comp * (one_minus_sqr_comp * conic.x - 0.3 * inv_det);
    v_cov2d.y += 2 * v_sqr_comp * (one_minus_sqr_comp * conic.y);
    v_cov2d.z += v_sqr_comp * (one_minus_sqr_comp * conic.z - 0.3 * inv_det);
}

// helper for applying R^T * p for a ROW MAJOR 4x3 matrix [R, t], ignoring t
inline __device__ float3 transform_4x3_rot_only_transposed(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[4] * p.y + mat[8] * p.z,
        mat[1] * p.x + mat[5] * p.y + mat[9] * p.z,
        mat[2] * p.x + mat[6] * p.y + mat[10] * p.z,
    };
    return out;
}

// helper for applying R^T * p for a ROW MAJOR 4x3 matrix [R, t], ignoring t
inline __device__ float3 transformVec4x3Transpose(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z,
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z,
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z,
    };
    return out;
}


// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(
    const float2 fxfy, const float3 p_view, const float2 pp
) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 p_proj = { p_view.x * rw, p_view.y * rw };
    float2 p_pix = { p_proj.x * fxfy.x + pp.x, p_proj.y * fxfy.y + pp.y };
    return p_pix;
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(
    const float2 fxfy, const float3 p_view, const float2 v_xy
) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 v_proj = { fxfy.x * v_xy.x, fxfy.y * v_xy.y };
    float3 v_view = {
        v_proj.x * rw, v_proj.y * rw, -(v_proj.x * p_view.x + v_proj.y * p_view.y) * rw * rw
    };
    return v_view;
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat_2d(const glm::vec4 quat) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y),
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x),
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}

inline __device__ float4
quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

inline __device__ glm::vec4
quat_to_rotmat_vjp_2d(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}

inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

inline __device__ glm::mat3
scale_to_mat_2d(const glm::vec3 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	S[2][2] = glob_scale * scale.z;
	return S;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3 &p_view, float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float sumf3(const float3& a) {
    return a.x + a.y + a.z;
}

inline __device__ float2 sqrtf2(const float2& a) {
    return make_float2(sqrtf(a.x), sqrtf(a.y));
}

inline __device__ float2 maxf2(float val, const float2& a) {
    return make_float2(fmaxf(val, a.x), fmaxf(val, a.y));
}

inline __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}