#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
void nvfp4_dequant_row_rvv(const uint8_t*, const uint8_t*, float, float*, size_t, float*);
#define ROWS 8
#define COLS 4096
int main(void){
    static uint8_t raw[ROWS][COLS/2], sc[ROWS][COLS/16];
    static float out[ROWS][COLS], ref[ROWS][COLS], scratch[COLS/16];
    float s2; FILE*f;
#define OPEN(p) f=fopen(p,"rb"); \
    if(!f){fprintf(stderr,"cannot open %s (run from testdata/, after nvfp4_rv)\n",p);return 2;}
    OPEN("w13_rows.bin")   if(fread(raw,1,sizeof raw,f)!=sizeof raw) return 2; fclose(f);
    OPEN("w13_scale.bin")  if(fread(sc,1,sizeof sc,f)!=sizeof sc)    return 2; fclose(f);
    OPEN("w13_scale2.bin") if(fread(&s2,4,1,f)!=1)                   return 2; fclose(f);
    OPEN("riscv_w.bin")    if(fread(ref,4,ROWS*COLS,f)!=ROWS*COLS)   return 2; fclose(f);

    for(int r=0;r<ROWS;r++)
        nvfp4_dequant_row_rvv(raw[r], sc[r], s2, out[r], COLS, scratch);

    size_t bad=0;
    for(int r=0;r<ROWS;r++) for(int c=0;c<COLS;c++){
        uint32_t a,b; memcpy(&a,&out[r][c],4); memcpy(&b,&ref[r][c],4);
        if(a!=b){ if(!bad) printf("  first mismatch r%d c%d %08x vs %08x\n",r,c,a,b); bad++; }
    }
    printf("VLEN=%ld bits | %d values | bit-exact vs scalar: %s (%zu mismatches)\n",
           (long)__riscv_vlenb()*8, ROWS*COLS, bad?"NO":"YES", bad);
    return bad!=0;
}
