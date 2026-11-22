/* NVFP4 -> FP32 dequant, RVV 1.0.
 *
 * Layout (as shipped in thinkingmachines/Inkling-Small-NVFP4):
 *   weight  U8, two E2M1 nibbles per byte, LOW nibble is the even element
 *   scale   F8_E4M3, one per 16 consecutive elements
 *   scale2  one F32 per expert
 *   value = E2M1 * e4m3(scale) * scale2
 *
 * No lookup table and no vrgather: E2M1 -> FP32 is pure bit assembly, so the
 * kernel is VLEN-agnostic and needs no element-count assumptions.
 *
 *   E2M1 nibble = s:1 e:2 m:1
 *     e != 0        ->  exp = e + 126, mantissa bit = m << 22
 *     e == 0, m = 1 ->  0.5  (exp 126, mantissa 0)   <- must be special-cased
 *     e == 0, m = 0 ->  +/-0.0
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>

static inline float e4m3_to_f32(uint8_t b){
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    float v = e ? ldexpf(1.f + m / 8.f, e - 7) : (m / 8.f) * 0.015625f;
    return s ? -v : v;
}


/* nibble (in the low 4 bits of each u32 lane) -> IEEE-754 binary32 bit pattern */
static inline vuint32m2_t e2m1_bits(vuint32m2_t n, size_t vl)
{
    vuint32m2_t s = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(n, 3, vl), 31, vl);
    vuint32m2_t e = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(n, 1, vl), 3, vl);
    vuint32m2_t m = __riscv_vand_vx_u32m2(n, 1, vl);

    /* normal case: exp = e + 126, mantissa bit = m << 22 */
    vuint32m2_t mag = __riscv_vor_vv_u32m2(
        __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(e, 126, vl), 23, vl),
        __riscv_vsll_vx_u32m2(m, 22, vl), vl);

    vbool16_t ez = __riscv_vmseq_vx_u32m2_b16(e, 0, vl);          /* e == 0 */
    /* e==0,m==1 -> 0.5 (0x3F000000); e==0,m==0 -> 0 */
    vuint32m2_t sub = __riscv_vmul_vx_u32m2(m, 0x3F000000u, vl);
    mag = __riscv_vmerge_vvm_u32m2(mag, sub, ez, vl);

    return __riscv_vor_vv_u32m2(s, mag, vl);
}

/* cols must be even; scales cover 16 elements each. */
void nvfp4_dequant_row_rvv(const uint8_t *raw, const uint8_t *sc8,
                           float scale2, float *out, size_t cols,
                           float *scratch /* >= cols/16 floats */)
{
    const size_t nblk = cols / 16;
    for (size_t i = 0; i < nblk; i++)          /* 1/16th of the data: scalar */
        scratch[i] = e4m3_to_f32(sc8[i]) * scale2;

    const size_t nbytes = cols / 2;            /* one byte -> two outputs */
    size_t i = 0;
    while (i < nbytes) {
        size_t vl = __riscv_vsetvl_e8mf2(nbytes - i);

        vuint8mf2_t b = __riscv_vle8_v_u8mf2(raw + i, vl);
        vuint32m2_t lo = __riscv_vzext_vf4_u32m2(__riscv_vand_vx_u8mf2(b, 0x0F, vl), vl);
        vuint32m2_t hi = __riscv_vzext_vf4_u32m2(__riscv_vsrl_vx_u8mf2(b, 4, vl), vl);

        /* one scale per 8 bytes; gather it as a float index load */
        vuint32m2_t byteidx = __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl), i, vl);
        vuint32m2_t soff    = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(byteidx, 3, vl), 2, vl);
        vfloat32m2_t s      = __riscv_vluxei32_v_f32m2(scratch, soff, vl);

        vfloat32m2_t vlo = __riscv_vreinterpret_v_u32m2_f32m2(e2m1_bits(lo, vl));
        vfloat32m2_t vhi = __riscv_vreinterpret_v_u32m2_f32m2(e2m1_bits(hi, vl));

        /* interleave the two nibbles back into element order */
        vfloat32m2x2_t pair = __riscv_vcreate_v_f32m2x2(
            __riscv_vfmul_vv_f32m2(vlo, s, vl),
            __riscv_vfmul_vv_f32m2(vhi, s, vl));
        __riscv_vsseg2e32_v_f32m2x2(out + 2 * i, pair, vl);

        i += vl;
    }
}
