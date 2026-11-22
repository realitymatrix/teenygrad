/* Portable NVFP4 dequant: E2M1 nibbles x E4M3 per-16 block scale x FP32 global. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static const float E2M1[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                              -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

static float e4m3_to_f32(uint8_t b){
    int s = (b>>7)&1, e = (b>>3)&0xF, m = b&7;
    float v = e ? ldexpf(1.f + m/8.f, e-7) : (m/8.f)*0.015625f;
    return s ? -v : v;
}

#define ROWS 8
#define NIB  2048            /* bytes per row */
#define COLS (NIB*2)         /* 4096 logical elements */
#define BLK  16
int main(void){
    static uint8_t raw[ROWS][NIB], sc[ROWS][COLS/BLK];
    static float out[ROWS][COLS];
    float s2;
    FILE *f;
    f=fopen("w13_rows.bin","rb");   if(!f||fread(raw,1,sizeof raw,f)!=sizeof raw) return 1; fclose(f);
    f=fopen("w13_scale.bin","rb");  if(!f||fread(sc,1,sizeof sc,f)!=sizeof sc)    return 1; fclose(f);
    f=fopen("w13_scale2.bin","rb"); if(!f||fread(&s2,4,1,f)!=1)                   return 1; fclose(f);

    static float bs[ROWS][COLS/BLK];
    for(int r=0;r<ROWS;r++)
        for(int k=0;k<COLS/BLK;k++) bs[r][k] = e4m3_to_f32(sc[r][k]) * s2;

    for(int r=0;r<ROWS;r++)
        for(int c=0;c<COLS;c++){
            uint8_t byte = raw[r][c>>1];
            uint8_t nib  = (c&1) ? (byte>>4) : (byte&0xF);
            out[r][c] = E2M1[nib] * bs[r][c/BLK];
        }

    f=fopen("riscv_w.bin","wb"); fwrite(out,4,ROWS*COLS,f); fclose(f);
    printf("dequantized %d values; w[0][0..3] = %.6f %.6f %.6f %.6f\n",
           ROWS*COLS, out[0][0], out[0][1], out[0][2], out[0][3]);
    return 0;
}
