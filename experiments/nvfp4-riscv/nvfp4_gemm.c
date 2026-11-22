/* Fused NVFP4 dequant + GEMM tile.
 *
 *   Y[m][b] = sum_k W[m][k] * X[b][k]      W is NVFP4, X and Y are FP32
 *
 * The point of fusing: a weight is unpacked ONCE per k and then reused across
 * all B columns of the tile, so the ~21 ops/element of dequant amortise by B
 * instead of being paid per multiply-add. Nothing is written to memory in
 * between, so the dequantised row never leaves the register file.
 *
 * Byte i of the weight row holds elements 2i (low nibble) and 2i+1 (high).
 * Rather than interleaving the two nibble vectors back into k order with a
 * segment store, we keep them apart and use a segment LOAD on X to split it
 * the same way: even lanes against the low nibbles, odd against the high.
 * Two accumulators per column, summed at the end.
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

static inline float e4m3_to_f32(uint8_t b){
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    float v = e ? ldexpf(1.f + m / 8.f, e - 7) : (m / 8.f) * 0.015625f;
    return s ? -v : v;
}

/* nibble in the low 4 bits of each u32 lane -> binary32 bit pattern */
static inline vuint32m1_t e2m1_bits(vuint32m1_t n, size_t vl)
{
    vuint32m1_t s = __riscv_vsll_vx_u32m1(__riscv_vsrl_vx_u32m1(n, 3, vl), 31, vl);
    vuint32m1_t e = __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(n, 1, vl), 3, vl);
    vuint32m1_t m = __riscv_vand_vx_u32m1(n, 1, vl);
    vuint32m1_t mag = __riscv_vor_vv_u32m1(
        __riscv_vsll_vx_u32m1(__riscv_vadd_vx_u32m1(e, 126, vl), 23, vl),
        __riscv_vsll_vx_u32m1(m, 22, vl), vl);
    vbool32_t ez = __riscv_vmseq_vx_u32m1_b32(e, 0, vl);
    mag = __riscv_vmerge_vvm_u32m1(mag, __riscv_vmul_vx_u32m1(m, 0x3F000000u, vl), ez, vl);
    return __riscv_vor_vv_u32m1(s, mag, vl);
}

/* RVV vector types are sizeless, so accumulators cannot go in an array: the
 * column loop is unrolled by macro and the width is a compile-time constant.
 * One instantiation per tile width. */
#define B1(M) M(0)
#define B2(M) M(0) M(1)
#define B4(M) M(0) M(1) M(2) M(3)
#define B8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

/* W: M rows of K/2 bytes.  SC: M rows of K/16 E4M3 scales.  X: B rows of K
 * floats, stride ldx.  Y: M*B floats, row-major [m][b].  scratch: >= K/16. */
#define GEN(BN, REP)                                                           \
void nvfp4_gemm_fused_b##BN(const uint8_t *W, const uint8_t *SC, float scale2, \
                            const float *X, float *Y,                          \
                            size_t M, size_t K, size_t ldx, float *scratch)    \
{                                                                              \
    const size_t nbytes = K / 2, nblk = K / 16, BW = BN;                       \
    const size_t vl0 = __riscv_vsetvlmax_e32m1();                              \
    for (size_t m = 0; m < M; m++) {                                           \
        const uint8_t *raw = W  + m * nbytes;                                  \
        const uint8_t *sc8 = SC + m * nblk;                                    \
        for (size_t i = 0; i < nblk; i++)                                      \
            scratch[i] = e4m3_to_f32(sc8[i]) * scale2;                         \
                                                                               \
        REP(DECL)                                                              \
        for (size_t i = 0; i < nbytes; ) {                                     \
            size_t vl = __riscv_vsetvl_e8mf4(nbytes - i);                      \
            vuint8mf4_t by = __riscv_vle8_v_u8mf4(raw + i, vl);                \
            vuint32m1_t lo = __riscv_vzext_vf4_u32m1(                          \
                                 __riscv_vand_vx_u8mf4(by, 0x0F, vl), vl);     \
            vuint32m1_t hi = __riscv_vzext_vf4_u32m1(                          \
                                 __riscv_vsrl_vx_u8mf4(by, 4, vl), vl);        \
            vuint32m1_t bidx = __riscv_vadd_vx_u32m1(                          \
                                   __riscv_vid_v_u32m1(vl), i, vl);            \
            vuint32m1_t soff = __riscv_vsll_vx_u32m1(                          \
                                   __riscv_vsrl_vx_u32m1(bidx, 3, vl), 2, vl); \
            vfloat32m1_t s = __riscv_vluxei32_v_f32m1(scratch, soff, vl);      \
            /* unpacked once; every column below reuses these two vectors */   \
            vfloat32m1_t we = __riscv_vfmul_vv_f32m1(                          \
                __riscv_vreinterpret_v_u32m1_f32m1(e2m1_bits(lo, vl)), s, vl); \
            vfloat32m1_t wo = __riscv_vfmul_vv_f32m1(                          \
                __riscv_vreinterpret_v_u32m1_f32m1(e2m1_bits(hi, vl)), s, vl); \
            REP(ACC)                                                           \
            i += vl;                                                           \
        }                                                                      \
        REP(STORE)                                                             \
    }                                                                          \
}

#define DECL(b)                                                                \
    vfloat32m1_t ae##b = __riscv_vfmv_v_f_f32m1(0.f, vl0);                     \
    vfloat32m1_t ao##b = __riscv_vfmv_v_f_f32m1(0.f, vl0);

#define ACC(b)                                                                 \
    {   vfloat32m1x2_t xp =                                                    \
            __riscv_vlseg2e32_v_f32m1x2(X + (b) * ldx + 2 * i, vl);            \
        ae##b = __riscv_vfmacc_vv_f32m1(ae##b, we,                             \
                    __riscv_vget_v_f32m1x2_f32m1(xp, 0), vl);                  \
        ao##b = __riscv_vfmacc_vv_f32m1(ao##b, wo,                             \
                    __riscv_vget_v_f32m1x2_f32m1(xp, 1), vl); }

#define STORE(b)                                                               \
    {   vfloat32m1_t sum = __riscv_vfadd_vv_f32m1(ae##b, ao##b, vl0);          \
        Y[m * BW + (b)] = __riscv_vfmv_f_s_f32m1_f32(                          \
            __riscv_vfredosum_vs_f32m1_f32m1(sum,                              \
                __riscv_vfmv_s_f_f32m1(0.f, 1), vl0)); }

GEN(1, B1)
GEN(2, B2)
GEN(4, B4)
GEN(8, B8)
