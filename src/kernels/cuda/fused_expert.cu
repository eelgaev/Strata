// Fused prefill expert products: see fused_expert.hpp.
#include "strata/kernels/fused_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
using namespace nvcuda;

// a block: 4 warps, TM tokens x BN weight rows, K in steps of BK; each warp a (TM/2) x 32 tile of 16x16 fragments
constexpr int TM = 64, BN = 64, BK = 64, LDS = BK + 8, LDE = BN + 4, NT = 128;
constexpr int SMEM = (TM + BN) * LDS * 2 > TM * LDE * 4 ? (TM + BN) * LDS * 2 : TM * LDE * 4;
using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

struct Tiles {
    const uint8_t* blob[kFusedExpertMax];
    int64_t row0[kFusedExpertMax];
    int ne[kFusedExpertMax], tile0[kFusedExpertMax + 1];
    int n;
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}
// SwiGLU as swiglu_il_kernel computes it (hf_sat included)
__device__ __forceinline__ __half silu_mul(float g, float u) {
    const float h = g / (1.0f + __expf(-g)) * u;
    return __float2half(isnan(h) ? h : fminf(fmaxf(h, -65504.0f), 65504.0f));
}
__device__ __forceinline__ void locate(const Tiles& g, int nrt, int& e, int& rt, int& ne, int64_t& row0) {
    const int t = (int) blockIdx.x;
    e = 0;
    while (e + 1 < g.n && g.tile0[e + 1] <= t) ++e;
    const int tt = (t - g.tile0[e]) / nrt;
    rt = (t - g.tile0[e]) % nrt;
    ne = min(TM, g.ne[e] - tt * TM);
    row0 = g.row0[e] + (int64_t) tt * TM;
}

// the X tile [TM][BK] through registers (fetched one step ahead); rows >= ne are zero
struct XRegs { uint4 v[TM * BK / 8 / NT]; };
__device__ __forceinline__ void fetch_x(XRegs& x, const __half* X, int64_t ld, int ne, int64_t k0) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8), c = (i % (BK / 8)) * 8;
        x.v[j] = r < ne ? *(const uint4*) (X + r * ld + k0 + c) : make_uint4(0, 0, 0, 0);
    }
}
__device__ __forceinline__ void commit_x(__half (*Xs)[LDS], const XRegs& x) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8), c = (i % (BK / 8)) * 8;
        *(uint4*) &Xs[r][c] = x.v[j];
    }
}
// the step's products in fresh fragments, then added to the running sum with ordinary FP32 adds: Volta's tensor
// cores truncate when they accumulate, and a 160-step chain (K = 2560) of truncations biases the sum toward zero
__device__ __forceinline__ void mma_tile(const __half (*Xs)[LDS], const __half (*Ws)[LDS], Acc (&tot)[TM / 32][2]) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2;
    Acc acc[TM / 32][2];
#pragma unroll
    for (int i = 0; i < TM / 32; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
#pragma unroll
    for (int k = 0; k < BK; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[TM / 32];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
        for (int i = 0; i < TM / 32; ++i) wmma::load_matrix_sync(a[i], &Xs[(TM / 2) * wm + 16 * i][k], LDS);
#pragma unroll
        for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(b[i], &Ws[32 * wn + 16 * i][k], LDS);
#pragma unroll
        for (int i = 0; i < TM / 32; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
#pragma unroll
    for (int i = 0; i < TM / 32; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int t = 0; t < tot[i][j].num_elements; ++t) tot[i][j].x[t] += acc[i][j].x[t];
}
__device__ __forceinline__ void store_acc(float (*E)[LDE], Acc (&acc)[TM / 32][2]) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2;
#pragma unroll
    for (int i = 0; i < TM / 32; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&E[(TM / 2) * wm + 16 * i][32 * wn + 16 * j], acc[i][j], LDE, wmma::mem_row_major);
}

// gate/up (Q4_K; interleaved row 2r = gate r, 2r + 1 = up r) and SwiGLU -> H
__global__ void __launch_bounds__(NT) gu_q4k_kernel(Tiles g, FusedExpertLayout L, const __half* __restrict__ X,
                                                    __half* __restrict__ H) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half (*Xs)[LDS] = (__half (*)[LDS]) sm;
    __half (*Ws)[LDS] = (__half (*)[LDS]) (sm + TM * LDS * 2);
    int e, rt, ne;
    int64_t row0;
    locate(g, (int) (2 * L.n_ff / BN), e, rt, ne, row0);
    Acc acc[TM / 32][2];
#pragma unroll
    for (int i = 0; i < TM / 32; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    // two threads per weight row: the low (sub-block 2 jj) and high (2 jj + 1) nibbles of a 64-value chunk
    const int wr = threadIdx.x / 2, hi = threadIdx.x % 2, R = rt * BN + wr;
    const uint8_t* rowp = g.blob[e] + ((R & 1) ? L.up_off : 0) + (size_t) (R >> 1) * L.gu_row;
    const __half* Xb = X + row0 * L.n_embd;
    XRegs xr;
    uint4 qr[2], hr;
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 256) * 144;
        const int jj = (int) (k0 % 256) / 64;
        hr = *(const uint4*) b;   // dm, scales[12]
        qr[0] = ((const uint4*) (b + 16 + 32 * jj))[0];
        qr[1] = ((const uint4*) (b + 16 + 32 * jj))[1];
    };
    auto commit_w = [&](int64_t k0) {
        const int jj = (int) (k0 % 256) / 64;
        const float2 dm = __half22float2(*(const __half2*) &hr.x);
        int s, m;
        scale_min_k4(2 * jj + hi, (const uint8_t*) &hr.y, s, m);
        // dq_q4_k's formulas: d1 = dall * sc, m1 = dmin * m, value d1 * q - m1
        const float d1 = dm.x * (uint8_t) s, m1 = dm.y * (uint8_t) m;
        const int sh = 4 * hi;
        __half* y = &Ws[wr][hi * 32];
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const uint8_t* q = (const uint8_t*) &qr[v];
            __align__(16) __half o[16];
#pragma unroll
            for (int l = 0; l < 16; ++l) o[l] = __float2half(d1 * ((q[l] >> sh) & 0xF) - m1);
            *(uint4*) &y[16 * v] = *(const uint4*) &o[0];
            *(uint4*) &y[16 * v + 8] = *(const uint4*) &o[8];
        }
    };
    fetch_x(xr, Xb, L.n_embd, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < L.n_embd; k0 += BK) {
        commit_x(Xs, xr);
        commit_w(k0);
        __syncthreads();
        if (k0 + BK < L.n_embd) { fetch_x(xr, Xb, L.n_embd, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * (BN / 2); i += NT) {
        const int r = i / (BN / 2), c = i % (BN / 2);
        if (r < ne) H[(row0 + r) * L.n_ff + rt * (BN / 2) + c] = silu_mul(E[r][2 * c], E[r][2 * c + 1]);
    }
}

// down (Q5_1 = 7 or Q8_0 = 8) -> D (FP32)
template<int DT>
__global__ void __launch_bounds__(NT) down_kernel(Tiles g, FusedExpertLayout L, const __half* __restrict__ H,
                                                  float* __restrict__ D) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half (*Xs)[LDS] = (__half (*)[LDS]) sm;
    __half (*Ws)[LDS] = (__half (*)[LDS]) (sm + TM * LDS * 2);
    int e, rt, ne;
    int64_t row0;
    locate(g, (int) (L.n_embd / BN), e, rt, ne, row0);
    Acc acc[TM / 32][2];
#pragma unroll
    for (int i = 0; i < TM / 32; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    // two threads per weight row, one 32-value block each
    const int wr = threadIdx.x / 2, hi = threadIdx.x % 2;
    constexpr int BB = DT == 7 ? 24 : 34;   // block bytes
    const uint8_t* rowp = g.blob[e] + L.down_off + (size_t) (rt * BN + wr) * L.d_row;
    const __half* Hb = H + row0 * L.n_ff;
    XRegs xr;
    uint2 w5[3];      // Q5_1: d, m, qh, qs[16] (8-byte aligned)
    uint16_t w8[17];  // Q8_0: d, qs[32] (2-byte aligned)
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 32 + hi) * BB;
        if constexpr (DT == 7) {
            w5[0] = ((const uint2*) b)[0]; w5[1] = ((const uint2*) b)[1]; w5[2] = ((const uint2*) b)[2];
        } else {
#pragma unroll
            for (int i = 0; i < 17; ++i) w8[i] = ((const uint16_t*) b)[i];
        }
    };
    auto commit_w = [&]() {
        __align__(16) __half y[32];
        if constexpr (DT == 7) {
            // dq_q5_1's formulas
            const float2 dm = __half22float2(*(const __half2*) &w5[0].x);
            const uint32_t qh = w5[0].y;
            const uint8_t* q = (const uint8_t*) &w5[1];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int xh_0 = ((qh >> (i + 0)) << 4) & 0x10, xh_1 = ((qh >> (i + 12))) & 0x10;
                y[i] = __float2half((float) ((q[i] & 0xf) | xh_0) * dm.x + dm.y);
                y[i + 16] = __float2half((float) ((q[i] >> 4) | xh_1) * dm.x + dm.y);
            }
        } else {
            const float d = __half2float(*(const __half*) &w8[0]);
            const int8_t* q = (const int8_t*) &w8[1];
#pragma unroll
            for (int i = 0; i < 32; ++i) y[i] = __float2half((float) q[i] * d);
        }
        __half* dst = &Ws[wr][hi * 32];
#pragma unroll
        for (int i = 0; i < 4; ++i) *(uint4*) &dst[8 * i] = *(const uint4*) &y[8 * i];
    };
    fetch_x(xr, Hb, L.n_ff, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < L.n_ff; k0 += BK) {
        commit_x(Xs, xr);
        commit_w();
        __syncthreads();
        if (k0 + BK < L.n_ff) { fetch_x(xr, Hb, L.n_ff, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * BN; i += NT) {
        const int r = i / BN, c = i % BN;
        if (r < ne) D[(row0 + r) * L.n_embd + rt * BN + c] = E[r][c];
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

bool fused_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    return gu_type == 12 && (d_type == 7 || d_type == 8) && n_embd % 256 == 0 && n_ff % BN == 0 && n_ff % BK == 0;
}

void fused_expert_run(const FusedExpertLayout& L, const FusedExpertGroup& in, const uint16_t* X, uint16_t* H, float* D,
                      void* stream) {
    if (in.n <= 0) return;
    if (!fused_expert_supported(L.gu_type, L.d_type, L.n_embd, L.n_ff) || in.n > kFusedExpertMax) {
        std::fprintf(stderr, "fused_expert_run: types %d/%d, %d experts\n", L.gu_type, L.d_type, in.n);
        std::exit(1);
    }
    static const bool once = [] {
        cudaFuncSetAttribute(gu_q4k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(down_kernel<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(down_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void) once;
    Tiles gu{}, dn{};
    int n = 0;
    for (int i = 0; i < in.n; ++i) {
        if (in.ne[i] <= 0) continue;
        gu.blob[n] = dn.blob[n] = in.blob[i];
        gu.row0[n] = dn.row0[n] = in.row0[i];
        gu.ne[n] = dn.ne[n] = in.ne[i];
        const int tt = (in.ne[i] + TM - 1) / TM;
        gu.tile0[n + 1] = gu.tile0[n] + tt * (int) (2 * L.n_ff / BN);
        dn.tile0[n + 1] = dn.tile0[n] + tt * (int) (L.n_embd / BN);
        ++n;
    }
    if (n == 0) return;
    gu.n = dn.n = n;
    const cudaStream_t s = (cudaStream_t) stream;
    gu_q4k_kernel<<<gu.tile0[n], NT, SMEM, s>>>(gu, L, (const __half*) X, (__half*) H);
    check("fused_expert gate/up");
    if (L.d_type == 7) down_kernel<7><<<dn.tile0[n], NT, SMEM, s>>>(dn, L, (const __half*) H, D);
    else down_kernel<8><<<dn.tile0[n], NT, SMEM, s>>>(dn, L, (const __half*) H, D);
    check("fused_expert down");
}

}  // namespace strata::kernels
