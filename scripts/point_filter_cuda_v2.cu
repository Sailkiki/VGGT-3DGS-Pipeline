// Multi-view geometric consistency filter for VGGT point cloud initialization.
// Projects each 3D candidate point to K neighbor views, verifies depth agreement
// between triangulated depth and VGGT-predicted depth, fuses multi-view color.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel 1: Multi-view geometric consistency scoring
__global__ void multiview_consistency_kernel_v2(
    const float* __restrict__ points_3d,       // [S, H, W, 3]  flat layout
    const float* __restrict__ depth_maps,       // [S, H, W]     flat layout
    const float* __restrict__ depth_conf,       // [S, H, W]
    const float* __restrict__ extrinsics,       // [S, 3, 4]     R|t world->cam
    const float* __restrict__ intrinsics,       // [S, 3, 3]     K matrix
    const int*    __restrict__ neighbor_indices,// [S, K]         best neighbors per frame
    const float* __restrict__ cam_centers,      // [S, 3]         precomputed -R^T * t
    const float* __restrict__ gradient_maps,    // [S, H, W]     precomputed Sobel magnitude
    float* __restrict__ consistency_score,      // [S*H*W]        output
    int S, int H, int W, int K,
    float consistency_thresh,
    float lambda_grad                           // gradient bonus weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * W;
    if (idx >= total) return;

    // Frame index and pixel coords
    int hw = H * W;
    int s = idx / hw;
    int residual = idx - s * hw;  // cheaper than %, only 1 mul + 1 sub
    int r = residual / W;
    int c = residual - r * W;

    // Read 3D point
    int pt_base = idx * 3;
    float px = points_3d[pt_base + 0];
    float py = points_3d[pt_base + 1];
    float pz = points_3d[pt_base + 2];

    if (pz <= 0.0f) {
        consistency_score[idx] = 0.0f;
        return;
    }

    float geo_sum = 0.0f;
    int valid_neighbors = 0;
    float grad_sum = 0.0f;

    // Use precomputed best-K neighbor indices for this frame
    const int* nbr_list = neighbor_indices + s * K;

    for (int n = 0; n < K; n++) {
        int t = nbr_list[n];
        if (t < 0 || t >= S || t == s) continue;

        // World → camera t
        const float* Rt = extrinsics + t * 12;
        float cam_x = Rt[0]*px + Rt[1]*py + Rt[2]*pz  + Rt[3];
        float cam_y = Rt[4]*px + Rt[5]*py + Rt[6]*pz  + Rt[7];
        float cam_z = Rt[8]*px + Rt[9]*py + Rt[10]*pz + Rt[11];

        if (cam_z <= 0.01f) continue;  // behind camera

        // Project to pixel (u, v)
        const float* Kt = intrinsics + t * 9;
        float u = Kt[0]*cam_x + Kt[1]*cam_y + Kt[2]*cam_z;
        float v = Kt[3]*cam_x + Kt[4]*cam_y + Kt[5]*cam_z;
        float w = Kt[6]*cam_x + Kt[7]*cam_y + Kt[8]*cam_z;

        u = u / w;
        v = v / w;

        // Bilinear interpolation for sub-pixel depth sampling
        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;

        if (u0 < 0 || u1 >= W || v0 < 0 || v1 >= H) continue;

        float wu = u - (float)u0;
        float wv = v - (float)v0;
        float w00 = (1.0f - wu) * (1.0f - wv);
        float w10 = wu * (1.0f - wv);
        float w01 = (1.0f - wu) * wv;
        float w11 = wu * wv;

        float* dm_t = (float*)depth_maps + t * hw;
        float d00 = dm_t[v0 * W + u0];
        float d10 = dm_t[v0 * W + u1];
        float d01 = dm_t[v1 * W + u0];
        float d11 = dm_t[v1 * W + u1];

        // Partial bilinear: require >= 2 valid corners (weight sum >= 0.5)
        float valid_w = 0.0f, depth_sum = 0.0f;
        if (d00 > 0.0f) { valid_w += w00; depth_sum += w00 * d00; }
        if (d10 > 0.0f) { valid_w += w10; depth_sum += w10 * d10; }
        if (d01 > 0.0f) { valid_w += w01; depth_sum += w01 * d01; }
        if (d11 > 0.0f) { valid_w += w11; depth_sum += w11 * d11; }
        if (valid_w < 0.5f) continue;

        float vggt_depth = depth_sum / valid_w;

        valid_neighbors++;

        // Soft geometric score: exponential decay with relative depth error
        float max_d = fmaxf(cam_z, vggt_depth);
        float rel_diff = fabsf(cam_z - vggt_depth) / max_d;
        geo_sum += expf(-rel_diff / consistency_thresh);

        // Gradient sampling at same sub-pixel location (partial bilinear)
        float* gm_t = (float*)gradient_maps + t * hw;
        float g00 = gm_t[v0 * W + u0];
        float g10 = gm_t[v0 * W + u1];
        float g01 = gm_t[v1 * W + u0];
        float g11 = gm_t[v1 * W + u1];

        float grad_sample = 0.0f;
        if (g00 >= 0.0f) grad_sample += w00 * g00;
        if (g10 >= 0.0f) grad_sample += w10 * g10;
        if (g01 >= 0.0f) grad_sample += w01 * g01;
        if (g11 >= 0.0f) grad_sample += w11 * g11;
        grad_sum += grad_sample;
    }

    // Score: soft geometric consistency + gradient bonus
    if (valid_neighbors > 0) {
        float geo_score = geo_sum / (float)valid_neighbors;
        float avg_grad = grad_sum / (float)valid_neighbors;
        consistency_score[idx] = geo_score + lambda_grad * avg_grad;
    } else {
        consistency_score[idx] = 0.0f;
    }
}


// Kernel 2: Weighted multi-view color fusion with view-angle weighting
__global__ void weighted_color_fusion_kernel_v2(
    const float* __restrict__ points_3d,       // [S, H, W, 3]
    const float* __restrict__ images,           // [S, H, W, 3]
    const float* __restrict__ extrinsics,       // [S, 3, 4]
    const float* __restrict__ intrinsics,       // [S, 3, 3]
    const float* __restrict__ cam_centers,      // [S, 3]  PRE-COMPUTED
    const int*    __restrict__ neighbor_indices,// [S, K]
    const float* __restrict__ consistency_score,// [S*H*W]
    float* __restrict__ fused_colors,           // [S*H*W, 3]
    float consistency_min,
    int S, int H, int W, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * W;
    if (idx >= total) return;

    int hw = H * W;
    int s = idx / hw;
    int residual = idx - s * hw;
    int r = residual / W;
    int c = residual - r * W;

    int out_base = idx * 3;

    float score = consistency_score[idx];
    if (score < consistency_min) {
        fused_colors[out_base + 0] = 0.5f;
        fused_colors[out_base + 1] = 0.5f;
        fused_colors[out_base + 2] = 0.5f;
        return;
    }

    // Read 3D point
    int pt_base = idx * 3;
    float px = points_3d[pt_base + 0];
    float py = points_3d[pt_base + 1];
    float pz = points_3d[pt_base + 2];

    // Source camera center (precomputed) → view direction to this point
    const float* ccs = cam_centers + s * 3;
    float vx = ccs[0] - px;
    float vy = ccs[1] - py;
    float vz = ccs[2] - pz;
    float dist = rsqrtf(vx*vx + vy*vy + vz*vz);
    vx *= dist; vy *= dist; vz *= dist;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_w = 0.0f;
    const int* nbr_list = neighbor_indices + s * K;

    // Sample source frame
    {
        int t = s;
        const float* Rt = extrinsics + t * 12;
        float cam_x = Rt[0]*px + Rt[1]*py + Rt[2]*pz  + Rt[3];
        float cam_y = Rt[4]*px + Rt[5]*py + Rt[6]*pz  + Rt[7];
        float cam_z = Rt[8]*px + Rt[9]*py + Rt[10]*pz + Rt[11];

        if (cam_z > 0.01f) {
            const float* Kt = intrinsics + t * 9;
            float u = Kt[0]*cam_x + Kt[1]*cam_y + Kt[2]*cam_z;
            float ve = Kt[3]*cam_x + Kt[4]*cam_y + Kt[5]*cam_z;
            float w = Kt[6]*cam_x + Kt[7]*cam_y + Kt[8]*cam_z;
            u /= w; ve /= w;

            // Bilinear color sampling (source frame)
            int u0 = (int)floorf(u), v0 = (int)floorf(ve);
            int u1 = u0 + 1, v1 = v0 + 1;
            if (u0 >= 0 && u1 < W && v0 >= 0 && v1 < H) {
                float wu = u - (float)u0, wv = ve - (float)v0;
                float w00 = (1.0f - wu) * (1.0f - wv);
                float w10 = wu * (1.0f - wv);
                float w01 = (1.0f - wu) * wv;
                float w11 = wu * wv;

                float* img_t = (float*)images + t * hw * 3;
                float cr = w00*img_t[v0*W*3 + u0*3 + 0] + w10*img_t[v0*W*3 + u1*3 + 0]
                         + w01*img_t[v1*W*3 + u0*3 + 0] + w11*img_t[v1*W*3 + u1*3 + 0];
                float cg = w00*img_t[v0*W*3 + u0*3 + 1] + w10*img_t[v0*W*3 + u1*3 + 1]
                         + w01*img_t[v1*W*3 + u0*3 + 1] + w11*img_t[v1*W*3 + u1*3 + 1];
                float cb = w00*img_t[v0*W*3 + u0*3 + 2] + w10*img_t[v0*W*3 + u1*3 + 2]
                         + w01*img_t[v1*W*3 + u0*3 + 2] + w11*img_t[v1*W*3 + u1*3 + 2];

                sum_r += cr;
                sum_g += cg;
                sum_b += cb;
                sum_w += 1.0f;
            }
        }
    }

    // Neighbor views (bilinear color sampling)
    for (int n = 0; n < K; n++) {
        int t = nbr_list[n];
        if (t < 0 || t >= S || t == s) continue;

        const float* Rt = extrinsics + t * 12;
        float cam_x = Rt[0]*px + Rt[1]*py + Rt[2]*pz  + Rt[3];
        float cam_y = Rt[4]*px + Rt[5]*py + Rt[6]*pz  + Rt[7];
        float cam_z = Rt[8]*px + Rt[9]*py + Rt[10]*pz + Rt[11];
        if (cam_z <= 0.01f) continue;

        const float* Kt = intrinsics + t * 9;
        float u = Kt[0]*cam_x + Kt[1]*cam_y + Kt[2]*cam_z;
        float ve = Kt[3]*cam_x + Kt[4]*cam_y + Kt[5]*cam_z;
        float w = Kt[6]*cam_x + Kt[7]*cam_y + Kt[8]*cam_z;
        u /= w; ve /= w;

        // Bilinear color sampling (neighbor view)
        int u0 = (int)floorf(u), v0 = (int)floorf(ve);
        int u1 = u0 + 1, v1 = v0 + 1;
        if (u0 < 0 || u1 >= W || v0 < 0 || v1 >= H) continue;

        float wu = u - (float)u0, wv = ve - (float)v0;
        float w00 = (1.0f - wu) * (1.0f - wv);
        float w10 = wu * (1.0f - wv);
        float w01 = (1.0f - wu) * wv;
        float w11 = wu * wv;

        // View angle weight via precomputed camera center
        const float* cct = cam_centers + t * 3;
        float tvx = cct[0] - px, tvy = cct[1] - py, tvz = cct[2] - pz;
        float td = rsqrtf(tvx*tvx + tvy*tvy + tvz*tvz);
        float dot = vx*tvx*td + vy*tvy*td + vz*tvz*td;
        dot = fmaxf(dot, 0.15f);

        float* img_t = (float*)images + t * hw * 3;
        float cr = w00*img_t[v0*W*3 + u0*3 + 0] + w10*img_t[v0*W*3 + u1*3 + 0]
                 + w01*img_t[v1*W*3 + u0*3 + 0] + w11*img_t[v1*W*3 + u1*3 + 0];
        float cg = w00*img_t[v0*W*3 + u0*3 + 1] + w10*img_t[v0*W*3 + u1*3 + 1]
                 + w01*img_t[v1*W*3 + u0*3 + 1] + w11*img_t[v1*W*3 + u1*3 + 1];
        float cb = w00*img_t[v0*W*3 + u0*3 + 2] + w10*img_t[v0*W*3 + u1*3 + 2]
                 + w01*img_t[v1*W*3 + u0*3 + 2] + w11*img_t[v1*W*3 + u1*3 + 2];

        sum_r += cr * dot;
        sum_g += cg * dot;
        sum_b += cb * dot;
        sum_w += dot;
    }

    if (sum_w > 0.0f) {
        fused_colors[out_base + 0] = sum_r / sum_w;
        fused_colors[out_base + 1] = sum_g / sum_w;
        fused_colors[out_base + 2] = sum_b / sum_w;
    } else {
        // fallback: source pixel
        int ib = s * hw * 3 + r * W * 3 + c * 3;
        fused_colors[out_base + 0] = images[ib+0];
        fused_colors[out_base + 1] = images[ib+1];
        fused_colors[out_base + 2] = images[ib+2];
    }
}


// Python bindings

torch::Tensor multiview_consistency_v2(
    torch::Tensor points_3d,          // [S, H, W, 3]
    torch::Tensor depth_maps,         // [S, H, W]
    torch::Tensor depth_conf,         // [S, H, W]
    torch::Tensor extrinsics,         // [S, 3, 4]
    torch::Tensor intrinsics,         // [S, 3, 3]
    torch::Tensor neighbor_indices,   // [S, K]  int64
    torch::Tensor cam_centers,        // [S, 3]  precomputed
    torch::Tensor gradient_maps,      // [S, H, W]  precomputed image gradient
    float consistency_thresh,
    float lambda_grad                 // gradient bonus weight
) {
    int S = points_3d.size(0);
    int H = points_3d.size(1);
    int W = points_3d.size(2);
    int K = neighbor_indices.size(1);
    int total = S * H * W;

    auto consistency_score = torch::empty({total},
        torch::TensorOptions().dtype(torch::kFloat32).device(points_3d.device()));

    const int threads = THREADS_PER_BLOCK;
    const int blocks = (total + threads - 1) / threads;

    multiview_consistency_kernel_v2<<<blocks, threads>>>(
        points_3d.data_ptr<float>(),
        depth_maps.data_ptr<float>(),
        depth_conf.data_ptr<float>(),
        extrinsics.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        neighbor_indices.data_ptr<int>(),
        cam_centers.data_ptr<float>(),
        gradient_maps.data_ptr<float>(),
        consistency_score.data_ptr<float>(),
        S, H, W, K,
        consistency_thresh,
        lambda_grad
    );

    return consistency_score;
}


torch::Tensor weighted_color_fusion_v2(
    torch::Tensor points_3d,
    torch::Tensor images,
    torch::Tensor extrinsics,
    torch::Tensor intrinsics,
    torch::Tensor cam_centers,           // [S, 3]
    torch::Tensor neighbor_indices,      // [S, K]
    torch::Tensor consistency_score,
    float consistency_min
) {
    int S = points_3d.size(0);
    int H = points_3d.size(1);
    int W = points_3d.size(2);
    int K = neighbor_indices.size(1);
    int total = S * H * W;

    auto fused_colors = torch::empty({total, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(points_3d.device()));

    const int threads = THREADS_PER_BLOCK;
    const int blocks = (total + threads - 1) / threads;

    weighted_color_fusion_kernel_v2<<<blocks, threads>>>(
        points_3d.data_ptr<float>(),
        images.data_ptr<float>(),
        extrinsics.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        cam_centers.data_ptr<float>(),
        neighbor_indices.data_ptr<int>(),
        consistency_score.data_ptr<float>(),
        fused_colors.data_ptr<float>(),
        consistency_min,
        S, H, W, K
    );

    return fused_colors;
}
