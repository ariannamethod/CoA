# ── CoA Makefile ───────────────────────────────────────────────────────────
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wno-unused-result -Wno-unused-function
LDFLAGS ?= -lm -lpthread

ifdef BLAS
  CFLAGS  += -DUSE_BLAS -DACCELERATE
  LDFLAGS += -framework Accelerate
endif

all: coa coa_v1_janus

# CUDA build target — links cuBLAS + cudart + ariannamethod GPU kernels.
# notorch.c includes notorch_cuda.h under #ifdef USE_CUDA; dispatch wiring
# in tape ops is the active port (commit-by-commit).
NVCC ?= nvcc
CUDA_CFLAGS  = -O3 -DUSE_CUDA -I.
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda

notorch_cuda.o: notorch_cuda.cu notorch_cuda.h
	$(NVCC) $(CUDA_CFLAGS) -c notorch_cuda.cu -o notorch_cuda.o

cuda: coa_v1_janus.c notorch.c notorch.h notorch_cuda.h notorch_cuda.cu loragrad.c loragrad.h
	$(NVCC) $(CUDA_CFLAGS) -c notorch_cuda.cu -o notorch_cuda.o
	$(CC) $(CFLAGS) -DUSE_CUDA -c notorch.c -o notorch_cuda_host.o
	$(CC) $(CFLAGS) -DUSE_CUDA -c loragrad.c -o loragrad_cuda.o
	$(CC) $(CFLAGS) -DUSE_CUDA -c coa_v1_janus.c -o coa_v1_janus_cuda.o
	$(CC) coa_v1_janus_cuda.o notorch_cuda_host.o notorch_cuda.o loragrad_cuda.o \
	      $(LDFLAGS) $(CUDA_LDFLAGS) -o coa_v1_janus_cuda

notorch.o: notorch.c notorch.h
	$(CC) $(CFLAGS) -c notorch.c -o notorch.o

loragrad.o: loragrad.c loragrad.h notorch.h
	$(CC) $(CFLAGS) -c loragrad.c -o loragrad.o

# v0 — vanilla MHA baseline (running 30K)
coa.o: coa.c notorch.h loragrad.h
	$(CC) $(CFLAGS) -c coa.c -o coa.o

coa: coa.o notorch.o loragrad.o
	$(CC) coa.o notorch.o loragrad.o $(LDFLAGS) -o coa

# v1 — Janus 3-attention (Content + RRPRAM-low-rank + Echo + SwiGLU + 1/3 blend)
coa_v1_janus.o: coa_v1_janus.c notorch.h loragrad.h
	$(CC) $(CFLAGS) -c coa_v1_janus.c -o coa_v1_janus.o

coa_v1_janus: coa_v1_janus.o notorch.o loragrad.o
	$(CC) coa_v1_janus.o notorch.o loragrad.o $(LDFLAGS) -o coa_v1_janus

# bpe_encode — pre-encoder tool (one-shot, output binary tokens)
bpe_encode: bpe_encode.c notorch.o
	$(CC) $(CFLAGS) bpe_encode.c notorch.o $(LDFLAGS) -o bpe_encode

# coa_infer — load saved .bin + run gen across multiple temps + top-k
coa_infer: coa_infer.c notorch.o
	$(CC) $(CFLAGS) coa_infer.c notorch.o $(LDFLAGS) -o coa_infer

run: coa
	./coa origin.txt

run-v1: coa_v1_janus
	./coa_v1_janus origin.txt

clean:
	rm -f coa coa.o coa_v1_janus coa_v1_janus.o notorch.o loragrad.o bpe_encode \
	      coa_v1_janus_cuda coa_v1_janus_cuda.o notorch_cuda.o notorch_cuda_host.o loragrad_cuda.o

.PHONY: all run run-v1 clean
