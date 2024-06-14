#include "forward.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>
#include <assert.h>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,                       // 高斯分佈的數量
    const float3* __restrict__ means3d,         // 高斯分佈的 3D 均值
    const float3* __restrict__ scales,          // 高斯分佈的尺度
    const float glob_scale,                     // 全局縮放比例
    const float4* __restrict__ quats,           // 旋轉四元數
    const float* __restrict__ viewmat,          // 視圖矩陣
    const float4 intrins,                       // 相機內部參數
    const dim3 img_size,                        // 圖像尺寸
    const dim3 tile_bounds,                     // 磚塊邊界
    const unsigned block_width,                 // 磚塊寬度
    const float clip_thresh,                    // 剪裁閾值
    float* __restrict__ covs3d,                 // 3D 協方差矩陣
    float2* __restrict__ xys,                   // 投影後的 2D 坐標
    float* __restrict__ depths,                 // 深度
    int* __restrict__ radii,                    // 半徑
    float3* __restrict__ conics,                // 圓錐
    float* __restrict__ compensation,           // 補償值
    int32_t* __restrict__ num_tiles_hit         // 命中的磚塊數量
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y,
    // p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    // printf("p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    // printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    // printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y,
    // quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d;
    float comp;
    project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy,
        cov2d, comp
    );
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix({fx, fy}, p_view, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max, block_width);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    compensation[idx] = comp;
    // printf("%d num_tiles_hit %.2f \n", idx, num_tiles_hit);

    // printf(
    //     "point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d
    //     %d, tile_max %d %d\n", idx, center.x, center.y, depths[idx],
    //     radii[idx], tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y
    // );
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max, block_width);
    // printf("point %d, %d radius, min %d %d, max %d %d\n", idx, radii[idx],
    // tile_min.x, tile_min.y, tile_max.x, tile_max.y);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    // printf("point %d starting at %d\n", idx, cur_idx);
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
__global__ void nd_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    int32_t pix_id = i * img_size.x + j;

    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t* id_batch = (int32_t*)s;
    float3* xy_opacity_batch = (float3*)&id_batch[block_size];
    float3* conic_batch = (float3*)&xy_opacity_batch[block_size];
    __half* color_out_batch = (__half*)&conic_batch[block_size];
    #pragma unroll
    for(int c = 0; c < channels; ++c)
        color_out_batch[block.thread_rank() * channels + c] = __float2half(0.f);

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    __half* pix_out = &color_out_batch[block.thread_rank() * channels];

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }
        // each thread fetch 1 gaussian from front to back

        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            #pragma unroll
            for (int c = 0; c < channels; ++c) {
                pix_out[c] = __hadd(pix_out[c], __float2half(colors[channels * g + c] * vis));
            }
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        #pragma unroll
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = __half2float(pix_out[c]) + T * background[c];
        }
    }
}

__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    int32_t pix_id = i * img_size.x + j;



    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];


//             printf("g: %d   c.x: %f     c.y: %f     c.z:%f\n", g, c.x, c.y, c.z);

            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ void project_cov3d_ewa(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 &cov2d,
    float &compensation
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs
    float c00 = cov[0][0], c11 = cov[1][1], c01 = cov[0][1];
    float det_orig = c00 * c11 - c01 * c01;
    cov2d.x = c00 + 0.3f;
    cov2d.y = c01;
    cov2d.z = c11 + 0.3f;
    float det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    compensation = std::sqrt(std::max(0.f, det_orig / det_blur));
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Compute a 2D-to-2D mapping matrix
__device__ void compute_transmat(
    const float3& p_world,
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* viewmatrix,
    const float* projmatrix,
    const int W,
    const int H,
    glm::mat3 &T,
    float3 &normal
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;

    // Make the geometry of 2D Gaussian as a Homogeneous transformation matrix
	// under the camera view, See Eq. (5) in 2DGS' paper.
    // center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(M[0], 0.0),
		glm::vec4(M[1], 0.0),
		glm::vec4(p_world.x, p_world.y, p_world.z, 1)
	);

    // 3d to 2d
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

    // projection into screen space, see Eq. (7) in 2DGS
	T = glm::transpose(splat2world) * world2ndc * ndc2pix;

	normal = transform_4x3_rot_only_transposed(viewmatrix, {M[2].x, M[2].y, M[2].z});

}


// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T,
	float2& point_image,
	float2 & extent
) {
	float3 T0 = {T[0][0], T[0][1], T[0][2]};
	float3 T1 = {T[1][0], T[1][1], T[1][2]};
	float3 T3 = {T[2][0], T[2][1], T[2][2]};

	// Compute AABB
	float3 temp_point = {1.0f, 1.0f, -1.0f};
	float distance = sumf3(T3 * T3 * temp_point);
	float3 f = (1 / distance) * temp_point;
	if (distance == 0.0) return false;

	point_image = {
		sumf3(f * T0 * T3),
		sumf3(f * T1 * T3)
	};

	float2 temp = {
		sumf3(f * T0 * T0),
		sumf3(f * T1 * T1)
	};
	float2 half_extend = point_image * point_image - temp;
	extent = sqrtf2(maxf2(1e-4, half_extend));
	return true;
}


// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel_2d(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* opacities,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float* __restrict__ transMats,
    float4* __restrict__ normal_opacity,
    int* __restrict__ radii,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    // printf("before clip near plane p_word %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y, p_world.z);
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y,
    // p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    // printf("after clip near plane p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    // printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    // printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y,
    // quat.z);

    // compute 2d to 2d matrix
    glm::mat3 T;
	float3 normal;

    compute_transmat(p_world, scale, glob_scale, quat, viewmat, projmat, img_size.x, img_size.y, T, normal);
    float3 *T_ptr = (float3*)transMats;

    T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
    T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
    T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};

    // Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, point_image, extent);
		if (!ok) return;
		radius = 3.0f * ceil(max(extent.x, extent.y));
	}
    // printf("%d radius %.2f \n", idx, radius);

    uint2 tile_min, tile_max;
    get_tile_bbox(point_image, radius, tile_bounds, tile_min, tile_max, block_width);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = point_image;
    normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};;
	num_tiles_hit[idx] = tile_area;
    // printf("%d num_tiles_hit %.2f \n", idx, num_tiles_hit);

}

__global__ void rasterize_forward_2d(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ transMats,
    const float3* __restrict__ colors,
    const float4* __restrict__ normal_opacity,
    float* __restrict__ final_T,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    float* __restrict__ out_others,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    float2 pixf = { (float)px, (float)py};
    int32_t pix_id = i * img_size.x + j;

    if (pix_id >= img_size.x * img_size.y) return;
    if (tile_id >= tile_bounds.x * tile_bounds.y) return;

    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    // normal opicaty
    __shared__ float4 collected_normal_opacity[MAX_BLOCK_SIZE];
    //xy point images
    __shared__ float2 collected_xy[MAX_BLOCK_SIZE];
    // T matrix
    __shared__ float3 collected_Tu[MAX_BLOCK_SIZE];
    __shared__ float3 collected_Tv[MAX_BLOCK_SIZE];
    __shared__ float3 collected_Tw[MAX_BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

#if RENDER_AXUTILITY
    // render axutility ouput
    float N[3] = {0};
    float D = {0};
    float M1 = {0};
    float M2 = {0};
    float distortion = {0};
    float median_depth = {0};
    float median_contributor = {-1};
#endif

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;

            collected_normal_opacity[block.thread_rank()] = normal_opacity[g_id];
            collected_xy[block.thread_rank()] = points_xy_image[g_id];
            collected_Tu[block.thread_rank()] = {transMats[9 * g_id+0], transMats[9 * g_id+1], transMats[9 * g_id+2]};
            collected_Tv[block.thread_rank()] = {transMats[9 * g_id+3], transMats[9 * g_id+4], transMats[9 * g_id+5]};
            collected_Tw[block.thread_rank()] = {transMats[9 * g_id+6], transMats[9 * g_id+7], transMats[9 * g_id+8]};
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            // First compute two homogeneous planes, See Eq. (8)
            const float2 xy = collected_xy[t];
            const float3 Tu = collected_Tu[t];
            const float3 Tv = collected_Tv[t];
            const float3 Tw = collected_Tw[t];
            float4 nor_o = collected_normal_opacity[t];

            float3 k = px * Tw - Tu;
            float3 l = py * Tw - Tv;
            float3 p = cross(k, l);
            if (p.z == 0.0) continue;
            assert(p.z != 0.0);
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

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];
#if RENDER_AXUTILITY
            // Render depth distortion map
            // Efficient implementation of distortion loss, see 2DGS' paper appendix.
            float A = 1-T;
            float m = far_n / (far_n - near_n) * (1 - near_n / depth);
            distortion += (m * m * A + M2 - 2 * m * M1) * vis;
            D  += depth * vis;
            M1 += m * vis;
            M2 += m * m * vis;

            if (T > 0.5) {
                median_depth = depth;
                median_contributor = batch_start + t;
            }
            for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * vis;
#endif
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_T[pix_id] = T;
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;

#if RENDER_AXUTILITY       
    final_index[pix_id + img_size.y * img_size.x] = median_contributor;
    final_T[pix_id + 1] = M1;
    final_T[pix_id + 2] = M2;
    out_others[pix_id + DEPTH_OFFSET * img_size.y * img_size.x] = D;
    out_others[pix_id + ALPHA_OFFSET * img_size.y * img_size.x] = 1 - T;
    for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * img_size.y * img_size.x] = N[ch];
    out_others[pix_id + MIDDEPTH_OFFSET * img_size.y * img_size.x] = median_depth;
    out_others[pix_id + DISTORTION_OFFSET * img_size.y * img_size.x] = distortion;
#endif
    // 调试输出
    // printf("pix_id: %d, final_index: %d, final_T[1]: %f, final_T[2]: %f\n", 
    //        pix_id, 
    //        final_index[pix_id + img_size.y * img_size.x], 
    //        final_T[pix_id + 1], 
    //        final_T[pix_id + 2]);

    // printf("pix_id: %d, out_others (D): %f, out_others (Alpha): %f\n", 
    //        pix_id, 
    //        out_others[pix_id + DEPTH_OFFSET * img_size.y * img_size.x], 
    //        out_others[pix_id + ALPHA_OFFSET * img_size.y * img_size.x]);

    // printf("pix_id: %d, out_others (median_depth): %f, out_others (distortion): %f\n", 
    //        pix_id, 
    //        out_others[pix_id + MIDDEPTH_OFFSET * img_size.y * img_size.x], 
    //        out_others[pix_id + DISTORTION_OFFSET * img_size.y * img_size.x]);
    
    }
}