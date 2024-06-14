#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <glm/glm.hpp>

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float* __restrict__ compensation,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    const float* __restrict__ v_compensation,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);


//kernel function for projecting each gaussian on device
__global__ void project_gaussians_backward_kernel_2d(
    const int num_points,
    const float3* __restrict__ means3d,
    const float* __restrict__ transMats,
    const glm::vec3* __restrict__ scales,
    const float glob_scale,
    const glm::vec4* __restrict__ rotations,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const dim3 img_size,
    const int* __restrict__ radii,
    // grad input
    const float* __restrict__ dL_dnormal3Ds,
    // grad output
    float* __restrict__ dL_dtransMats,
    float3* __restrict__ dL_dmean2Ds,
    glm::vec3* __restrict__ dL_dmean3Ds,
    glm::vec2* __restrict__ dL_dscales,
    glm::vec4* __restrict__ dL_drots
);

// compute jacobians of output image wrt binned and sorted gaussians
__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float* __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
);

__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);

__global__ void project_gaussians_backward_kernel_2d(
    const int num_points,
    const float3* __restrict__ means3d,
    const float* __restrict__ transMats,
    const glm::vec3* __restrict__ scales,
    const float glob_scale,
    const glm::vec4* __restrict__ rotations,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const dim3 img_size,
    const int* __restrict__ radii,
    // grad input
    const float* __restrict__ dL_dnormal3Ds,
    // grad output
    float* __restrict__ dL_dtransMats,
    float3* __restrict__ dL_dmean2Ds,
    glm::vec3* __restrict__ dL_dmean3Ds,
    glm::vec2* __restrict__ dL_dscales,
    glm::vec4* __restrict__ dL_drots
);

__global__ void rasterize_backward_kernel_2d(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ points_xy_image,
    const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
    const float3* __restrict__ rgbs,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    // grad input
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    // grad output
    float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
    float* __restrict__ dL_dopacity,
    float3* __restrict__ v_rgb
);