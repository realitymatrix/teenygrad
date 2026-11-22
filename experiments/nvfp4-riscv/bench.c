#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <riscv_vector.h>
void nvfp4_dequant_row_rvv(const uint8_t*,const uint8_t*,float,float*,size_t,float*);
#define COLS 4096
static const float E2M1[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static float e4m3(uint8_t b){int s=(b>>7)&1,e=(b>>3)&0xF,m=b&7;
    float v=e?ldexpf(1.f+m/8.f,e-7):(m/8.f)*0.015625f; return s?-v:v;}
static void scalar(const uint8_t*raw,const uint8_t*sc,float s2,float*out,size_t cols,float*bs){
    for(size_t k=0;k<cols/16;k++) bs[k]=e4m3(sc[k])*s2;
    for(size_t c=0;c<cols;c++){uint8_t by=raw[c>>1];
        out[c]=E2M1[(c&1)?(by>>4):(by&0xF)]*bs[c/16];}}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
    int iters = argc>1?atoi(argv[1]):2000;
    static uint8_t raw[COLS/2],sc[COLS/16]; static float out[COLS],bs[COLS/16];
    FILE*f=fopen("w13_rows.bin","rb"); if(fread(raw,1,sizeof raw,f)!=sizeof raw)return 1; fclose(f);
    f=fopen("w13_scale.bin","rb"); if(fread(sc,1,sizeof sc,f)!=sizeof sc)return 1; fclose(f);
    float s2; f=fopen("w13_scale2.bin","rb"); if(fread(&s2,4,1,f)!=1)return 1; fclose(f);
    double t0=now(); for(int i=0;i<iters;i++) scalar(raw,sc,s2,out,COLS,bs); double ts=now()-t0;
    t0=now(); for(int i=0;i<iters;i++) nvfp4_dequant_row_rvv(raw,sc,s2,out,COLS,bs); double tv=now()-t0;
    double n=(double)iters*COLS;
    printf("VLEN=%4ld  scalar %7.1f Melem/s   RVV %7.1f Melem/s   speedup %.2fx\n",
           (long)__riscv_vlenb()*8, n/ts/1e6, n/tv/1e6, ts/tv);
    return 0;}
