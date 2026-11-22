# NVFP4 dequant on RISC-V. Requires riscv64-linux-gnu-gcc and qemu-riscv64-static.
CC      := riscv64-linux-gnu-gcc
CFLAGS  := -O2 -static
VFLAGS  := -march=rv64gcv
QEMU    := qemu-riscv64-static
VLENS   := 128 256 512 1024

all: nvfp4_rv nvfp4_rvv_test nvfp4_bench nvfp4_gemm_test

nvfp4_rv: nvfp4.c                      # portable scalar reference
	$(CC) $(CFLAGS) -o $@ $< -lm
nvfp4_rvv_test: harness.c nvfp4_rvv.c  # RVV vs scalar, bit-exactness
	$(CC) $(VFLAGS) $(CFLAGS) -o $@ $^ -lm
nvfp4_bench: bench.c nvfp4_rvv.c       # throughput; run this on real hardware
	$(CC) $(VFLAGS) $(CFLAGS) -o $@ $^ -lm
nvfp4_gemm_test: gemm_test.c nvfp4_gemm.c  # fused dequant+GEMM: accuracy and amortisation
	$(CC) $(VFLAGS) $(CFLAGS) -o $@ $^ -lm

# Full check: numpy -> scalar riscv -> RVV at every VLEN. Run from this dir.
check: all
	cd testdata && python3 -I ../ref.py >/dev/null
	cd testdata && $(QEMU) ../nvfp4_rv
	cd testdata && python3 -I -c "import numpy as np; \
a=np.load('ref_w.npy').astype(np.float32).ravel(); b=np.fromfile('riscv_w.bin',np.float32); \
print('numpy vs scalar riscv bit-exact:', np.array_equal(a.view(np.uint32),b.view(np.uint32)))"
	@for v in $(VLENS); do \
	  (cd testdata && $(QEMU) -cpu rv64,v=true,vlen=$$v,elen=64 ../nvfp4_rvv_test); done
	cd testdata && $(QEMU) -cpu rv64,v=true,vlen=256,elen=64 ../nvfp4_gemm_test 50

bench: nvfp4_bench
	cd testdata && $(QEMU) -cpu rv64,v=true,vlen=256,elen=64 ../nvfp4_bench 3000

gemm: nvfp4_gemm_test
	cd testdata && $(QEMU) -cpu rv64,v=true,vlen=256,elen=64 ../nvfp4_gemm_test 150

clean:
	rm -f nvfp4_rv nvfp4_rvv_test nvfp4_bench nvfp4_gemm_test testdata/ref_w.npy testdata/riscv_w.bin
.PHONY: all check bench gemm clean
