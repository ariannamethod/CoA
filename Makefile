# ── CoA Makefile ───────────────────────────────────────────────────────────
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wno-unused-result -Wno-unused-function
LDFLAGS ?= -lm -lpthread

ifdef BLAS
  CFLAGS  += -DUSE_BLAS -DACCELERATE
  LDFLAGS += -framework Accelerate
endif

all: coa coa_v1_janus

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

run: coa
	./coa origin.txt

run-v1: coa_v1_janus
	./coa_v1_janus origin.txt

clean:
	rm -f coa coa.o coa_v1_janus coa_v1_janus.o notorch.o loragrad.o bpe_encode

.PHONY: all run run-v1 clean
