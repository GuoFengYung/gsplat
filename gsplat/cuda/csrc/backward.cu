#include "backward.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}
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
) {
    auto block = cg::this_thread_block();
    const int tr = block.thread_rank();
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);
    // which gaussians get gradients for this pixel
    const int2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    const float v_out_alpha = v_output_alpha[pix_id];
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    
    extern __shared__ half workspace[];

    half *S = (half*)(&workspace[channels * tr]);
    #pragma unroll
    for(int c=0; c<channels; ++c){
        S[c] = __float2half(0.f);
    }
    const int bin_final = inside ? final_index[pix_id] : 0;
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int idx = warp_bin_final - 1; idx >= range.x; --idx) {
        int valid = inside && idx < bin_final;
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        valid &= (sigma >= 0.f);
        const float opac = opacities[g];
        const float vis = __expf(-sigma);
        const float alpha = min(0.99f, opac * vis);
        valid &= (alpha >= 1.f / 255.f);
        if(!warp.any(valid)){
            continue;
        }
        float v_alpha = 0.f;
        float3 v_conic_local = {0.f, 0.f, 0.f};
        float2 v_xy_local = {0.f, 0.f};
        float2 v_xy_abs_local = {0.f, 0.f};
        float v_opacity_local = 0.f;
        if(valid){
            // compute the current T for this gaussian
            const float ra = 1.f / (1.f - alpha);
            T *= ra;
            // update v_rgb for this gaussian
            const float fac = alpha * T;
            for (int c = 0; c < channels; ++c) {
                // gradient wrt rgb
                atomicAdd(&(v_rgb[channels * g + c]), fac * v_out[c]);
                // contribution from this pixel
                v_alpha += (rgbs[channels * g + c] * T - __half2float(S[c]) * ra) * v_out[c];
                // contribution from background pixel
                v_alpha += -T_final * ra * background[c] * v_out[c];
                // update the running sum
                S[c] = __hadd(S[c], __float2half(rgbs[channels * g + c] * fac));
            }
            v_alpha += T_final * ra * v_out_alpha;
            const float v_sigma = -opac * vis * v_alpha;
            v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                             v_sigma * delta.x * delta.y,
                             0.5f * v_sigma * delta.y * delta.y};
            v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                          v_sigma * (conic.y * delta.x + conic.z * delta.y)};
            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
            v_opacity_local = vis * v_alpha;
        }
        warpSum3(v_conic_local, warp);
        warpSum2(v_xy_local, warp);
        warpSum2(v_xy_abs_local, warp);
        warpSum(v_opacity_local, warp);
        if (warp.thread_rank() == 0) {
            float* v_conic_ptr = (float*)(v_conic);
            float* v_xy_ptr = (float*)(v_xy);
            float* v_xy_abs_ptr = (float*)(v_xy_abs);
            atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
            atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
            atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
            atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
            atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
            atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
            atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
            atomicAdd(v_opacity + g, v_opacity_local);
        }
    }
}

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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                 v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y), 
                                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum2(v_xy_abs_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_xy_abs_ptr = (float*)(v_xy_abs);
                atomicAdd(v_xy_abs_ptr + 2*g + 0, v_xy_abs_local.x);
                atomicAdd(v_xy_abs_ptr + 2*g + 1, v_xy_abs_local.y);
                
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}

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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
    float fx = intrins.x;
    float fy = intrins.y;
    float3 p_view = transform_4x3(viewmat, p_world);
    // get v_mean3d from v_xy
    v_mean3d[idx] = transform_4x3_rot_only_transposed(
        viewmat,
        project_pix_vjp({fx, fy}, p_view, v_xy[idx]));

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    v_mean3d[idx].x += viewmat[8] * v_z;
    v_mean3d[idx].y += viewmat[9] * v_z;
    v_mean3d[idx].z += viewmat[10] * v_z;

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    cov2d_to_compensation_vjp(compensation[idx], conics[idx], v_compensation[idx], v_cov2d[idx]);
    // get v_cov3d (and v_mean3d contribution)
    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        v_cov2d[idx],
        v_mean3d[idx],
        &(v_cov3d[6 * idx])
    );
    // get v_scale and v_quat
    scale_rot_to_cov3d_vjp(
        scales[idx],
        glob_scale,
        quats[idx],
        &(v_cov3d[6 * idx]),
        v_scale[idx],
        v_quat[idx]
    );
}

// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float3& __restrict__ v_cov2d,
    float3& __restrict__ v_mean3d,
    float* __restrict__ v_cov3d
) {
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    // clang-format off
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    glm::mat3 J = glm::mat3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    glm::mat3 T = J * W;
    glm::mat3 Tt = glm::transpose(T);
    glm::mat3 Vt = glm::transpose(V);
    glm::mat3 v_V = Tt * v_cov * T;
    glm::mat3 v_T = v_cov * T * Vt + glm::transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    glm::mat3 v_J = v_T * glm::transpose(W);
    float rz3 = rz2 * rz;
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d.x += (float)glm::dot(v_t, W[0]);
    v_mean3d.y += (float)glm::dot(v_t, W[1]);
    v_mean3d.z += (float)glm::dot(v_t, W[2]);
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* __restrict__ v_cov3d,
    float3& __restrict__ v_scale,
    float4& __restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]) * glob_scale;
    v_scale.y = (float)glm::dot(R[1], v_M[1]) * glob_scale;
    v_scale.z = (float)glm::dot(R[2], v_M[2]) * glob_scale;

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}

__device__ void compute_transmat_aabb(
	int idx, 
    const float3* __restrict__ means3d,
    const float* Ts_precomp,
	const glm::vec3* scales, 
    const float glob_scale,
	const glm::vec4* rots, 
    const float* viewmatrix,
    const float* projmatrix,
    const int W,
    const int H,
    const float3* dL_dnormals,
    const float3* dL_dmean2Ds,
    float* dL_dTs, 
	glm::vec3* dL_dmeans,
	glm::vec2* dL_dscales,
    glm::vec4* dL_drots
    ) {

	glm::mat3 T;
	float3 normal;
    glm::mat3x4 P;
    glm::mat3 R;
	glm::vec4 rot;
	glm::vec3 scale;

    // Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		float3 p_orig = means3d[idx];
        rot = rots[idx];
		scale = scales[idx];
        
		glm::mat3 R = quat_to_rotmat_2d(rot);
		glm::mat3 S = scale_to_mat_2d(scale, glob_scale);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transform_4x3_rot_only_transposed(viewmatrix, {L[2].x, L[2].y, L[2].z});
	}

    // Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		const float distance = T[2].x * T[2].x + T[2].y * T[2].y - T[2].z * T[2].z;
		const float f = 1 / (distance);
		const float dpx_dT00 =  f * T[2].x;
		const float dpx_dT01 =  f * T[2].y;
		const float dpx_dT02 = -f * T[2].z;
		const float dpy_dT10 =  f * T[2].x;
		const float dpy_dT11 =  f * T[2].y;
		const float dpy_dT12 = -f * T[2].z;
		const float dpx_dT30 =  T[0].x * (f - 2 * f * f * T[2].x * T[2].x);
		const float dpx_dT31 =  T[0].y * (f - 2 * f * f * T[2].y * T[2].y);
		const float dpx_dT32 = -T[0].z * (f + 2 * f * f * T[2].z * T[2].z);
		const float dpy_dT30 =  T[1].x * (f - 2 * f * f * T[2].x * T[2].x);
		const float dpy_dT31 =  T[1].y * (f - 2 * f * f * T[2].y * T[2].y);
		const float dpy_dT32 = -T[1].z * (f + 2 * f * f * T[2].z * T[2].z);

		dL_dT[0].x += dL_dmean2D.x * dpx_dT00;
		dL_dT[0].y += dL_dmean2D.x * dpx_dT01;
		dL_dT[0].z += dL_dmean2D.x * dpx_dT02;
		dL_dT[1].x += dL_dmean2D.y * dpy_dT10;
		dL_dT[1].y += dL_dmean2D.y * dpy_dT11;
		dL_dT[1].z += dL_dmean2D.y * dpy_dT12;
		dL_dT[2].x += dL_dmean2D.x * dpx_dT30 + dL_dmean2D.y * dpy_dT30;
		dL_dT[2].y += dL_dmean2D.x * dpx_dT31 + dL_dmean2D.y * dpy_dT31;
		dL_dT[2].z += dL_dmean2D.x * dpx_dT32 + dL_dmean2D.y * dpy_dT32;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
    if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	
    float3 dL_dtn = transformVec4x3Transpose(viewmatrix, dL_dnormals[idx]);

    	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp_2d(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);

}


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
) {
//     printf("is right");
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) return;


    const float* Ts_precomp = transMats;
    const int W = int(img_size.x);
    const int H = int(img_size.y);
//     printf("%d idx %d %d \n", idx, H, W);

    compute_transmat_aabb(
        idx,
        means3d,
        Ts_precomp,
        scales,
        glob_scale,
        rotations,
        viewmat,
        projmat,
        W,
        H,
        (float3*)dL_dnormal3Ds,
        dL_dmean2Ds,
        dL_dtransMats,
        dL_dmean3Ds,
        dL_dscales,
        dL_drots
    );

    float depth = transMats[idx * 9 + 8];
    dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
    dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


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
    // const float* __restrict__ dL_depths,
    const float* __restrict__ v_output_alpha,
    // grad output
    float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
    float* __restrict__ dL_dopacity,
    float3* __restrict__ v_rgb
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float4 collected_normal_opacity[MAX_BLOCK_SIZE];
    __shared__ float2 collected_xy[MAX_BLOCK_SIZE];
	__shared__ float3 collected_Tu[MAX_BLOCK_SIZE];
	__shared__ float3 collected_Tv[MAX_BLOCK_SIZE];
	__shared__ float3 collected_Tw[MAX_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    // const float v_out_alpha = v_output_alpha[pix_id];
    // printf("pix_id: %d v_output_alpha: %f\n", pix_id, v_output_alpha);

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? final_index[pix_id + img_size.x * img_size.y] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = v_output_alpha[DEPTH_OFFSET * img_size.x * img_size.y + pix_id];
        // printf("dL_ddepth: %f\n", dL_ddepth);
		dL_daccum = v_output_alpha[ALPHA_OFFSET * img_size.x * img_size.y + pix_id];
		dL_dreg = v_output_alpha[DISTORTION_OFFSET * img_size.x * img_size.y + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = v_output_alpha[(NORMAL_OFFSET + i) * img_size.x * img_size.y + pix_id];

		dL_dmedian_depth = v_output_alpha[MIDDEPTH_OFFSET * img_size.x * img_size.y + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + img_size.x * img_size.y] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * img_size.x * img_size.y] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            rgbs_batch[tr] = rgbs[g_id];
			collected_xy[block.thread_rank()] = points_xy_image[g_id];
            collected_normal_opacity[block.thread_rank()] = normal_opacity[g_id];
            collected_Tu[block.thread_rank()] = {transMats[9 * g_id+0], transMats[9 * g_id+1], transMats[9 * g_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * g_id+3], transMats[9 * g_id+4], transMats[9 * g_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * g_id+6], transMats[9 * g_id+7], transMats[9 * g_id+8]};
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }

			const float2 xy = collected_xy[t];
			const float3 Tu = collected_Tu[t];
			const float3 Tv = collected_Tv[t];
			const float3 Tw = collected_Tw[t];
            float4 nor_o = collected_normal_opacity[t];

			float3 k = px * Tw - Tu;
			float3 l = py * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
            float2 s = {p.x / p.z, p.y / p.z};
            float rho3d = (s.x * s.x + s.y * s.y);
            float2 d = {xy.x - px, xy.y - py};
            float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

            // compute intersection and depth
            float rho = min(rho3d, rho2d);
            float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
            if (depth < near_n) continue;
            float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
            float opa = nor_o.w;

            float power = -0.5f * rho;
            if (power > 0.0f) continue;

            const float alpha = min(0.999f, opa * __expf(power));
            if (alpha < 0.f || alpha < 1.f / 255.f) continue;
            
            float dL_dalpha = 0.0f;
            const int global_id = id_batch[t];
            // const float T = T / (1.f - alpha);
            // const float dchannel_dcolor = alpha * T;
			// const float vis = alpha * T;

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }

            // Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * dL_ddepth;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;
                // Helpful reusable temporary variables
			    const float dL_dG = nor_o.w * v_alpha;
                float v_depth = 0.0f;
                float dL_dz = 0.0f;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif
                if (rho3d <= rho2d) {
                    // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
                    const float2 dL_ds = {
                        dL_dG * -power * s.x + dL_dz * Tw.x,
                        dL_dG * -power * s.y + dL_dz * Tw.y
                    };
                    const float3 dz_dTw = {s.x, s.y, 1.0};
                    const float dsx_pz = dL_ds.x / p.z;
                    const float dsy_pz = dL_ds.y / p.z;
                    const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
                    const float3 dL_dk = cross(l, dL_dp);
                    const float3 dL_dl = cross(dL_dp, k);

                    const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
                    const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
                    const float3 dL_dTw = {
                        px * dL_dk.x + py * dL_dl.x + dL_dz * dz_dTw.x, 
                        px * dL_dk.y + py * dL_dl.y + dL_dz * dz_dTw.y, 
                        px * dL_dk.z + py * dL_dl.z + dL_dz * dz_dTw.z};

                    // Update gradients w.r.t. 3D covariance (3x3 matrix)
                    atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
                    atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
                }else {
                    // // Update gradients w.r.t. center of Gaussian 2D mean position
                    const float dG_ddelx = -power * FilterInvSquare * d.x;
                    const float dG_ddely = -power * FilterInvSquare * d.y;
                    atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
                    atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
                    atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			    }   

                // Update gradients w.r.t. opacity of the Gaussian
                atomicAdd(&(dL_dopacity[global_id]), power * dL_dalpha);
            }
            warpSum3(v_rgb_local, warp);
            if (warp.thread_rank() == 0) {
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*global_id + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*global_id + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*global_id + 2, v_rgb_local.z);
            }
        }
    }
}
