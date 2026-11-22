#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>
#define M 8
#define K 4096
void nvfp4_gemm_fused_b1(const uint8_t*,const uint8_t*,float,const float*,float*,size_t,size_t,size_t,float*);
void nvfp4_gemm_fused_b2(const uint8_t*,const uint8_t*,float,const float*,float*,size_t,size_t,size_t,float*);
void nvfp4_gemm_fused_b4(const uint8_t*,const uint8_t*,float,const float*,float*,size_t,size_t,size_t,float*);
void nvfp4_gemm_fused_b8(const uint8_t*,const uint8_t*,float,const float*,float*,size_t,size_t,size_t,float*);
static const float E2M1[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static float e4m3(uint8_t b){int s=(b>>7)&1,e=(b>>3)&0xF,m=b&7;
    float v=e?ldexpf(1.f+m/8.f,e-7):(m/8.f)*0.015625f; return s?-v:v;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static uint8_t raw[M][K/2], sc[M][K/16];
static float s2, X[8][K], Y[M*8], R[M*8], wbuf[K], bs[K/16];

/* unfused: dequantise the whole row to memory, then multiply */
static void unfused(size_t B){
    for(size_t m=0;m<M;m++){
        for(size_t k=0;k<K/16;k++) bs[k]=e4m3(sc[m][k])*s2;
        for(size_t k=0;k<K;k++){uint8_t by=raw[m][k>>1];
            wbuf[k]=E2M1[(k&1)?(by>>4):(by&0xF)]*bs[k/16];}
        for(size_t b=0;b<B;b++){float a=0; for(size_t k=0;k<K;k++) a+=wbuf[k]*X[b][k];
            R[m*B+b]=a;}}
}
int main(int argc,char**argv){
    int iters=argc>1?atoi(argv[1]):200;
    FILE*f;
#define OPEN(p) f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s (run from testdata/)\n",p);return 2;}
    OPEN("w13_rows.bin")   if(fread(raw,1,sizeof raw,f)!=sizeof raw) return 2; fclose(f);
    OPEN("w13_scale.bin")  if(fread(sc,1,sizeof sc,f)!=sizeof sc)   return 2; fclose(f);
    OPEN("w13_scale2.bin") if(fread(&s2,4,1,f)!=1)                  return 2; fclose(f);
    uint32_t st=12345; for(int b=0;b<8;b++) for(int k=0;k<K;k++){
        st=st*1664525u+1013904223u; X[b][k]=(float)((int32_t)(st>>8)%2001-1000)/1000.f; }

    printf("VLEN=%ld  M=%d K=%d\n",(long)__riscv_vlenb()*8,M,K);
    size_t widths[4]={1,2,4,8};
    for(int w=0;w<4;w++){
        size_t B=widths[w];
        switch(B){
          case 1: nvfp4_gemm_fused_b1(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 2: nvfp4_gemm_fused_b2(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 4: nvfp4_gemm_fused_b4(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 8: nvfp4_gemm_fused_b8(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break; }
        unfused(B);
        double worst=0; for(size_t i=0;i<M*B;i++){
            double d=fabs(Y[i]-R[i])/(fabs(R[i])+1e-9); if(d>worst) worst=d; }

        double t0=now(); for(int i=0;i<iters;i++) unfused(B); double tu=now()-t0;
        t0=now(); for(int i=0;i<iters;i++){ switch(B){
          case 1: nvfp4_gemm_fused_b1(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 2: nvfp4_gemm_fused_b2(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 4: nvfp4_gemm_fused_b4(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break;
          case 8: nvfp4_gemm_fused_b8(raw[0],sc[0],s2,X[0],Y,M,K,K,bs); break; } }
        double tf=now()-t0;
        double mac=(double)iters*M*K*B;
        printf("  B=%zu  rel.err %.2e | unfused %6.1f  fused %6.1f MMAC/s | %.2fx\n",
               B, worst, mac/tu/1e6, mac/tf/1e6, tu/tf);
    }
    return 0;
}
