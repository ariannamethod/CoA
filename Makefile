# ── CoA Makefile ───────────────────────────────────────────────────────────
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wno-unused-result -Wno-unused-function
LDFLAGS ?= -lm -lpthread

ifdef BLAS
  CFLAGS  += -DUSE_BLAS -DACCELERATE
  LDFLAGS += -framework Accelerate
endif

all: coa

notorch.o: notorch.c notorch.h
	$(CC) $(CFLAGS) -c notorch.c -o notorch.o

loragrad.o: loragrad.c loragrad.h notorch.h
	$(CC) $(CFLAGS) -c loragrad.c -o loragrad.o

coa.o: coa.c notorch.h loragrad.h
	$(CC) $(CFLAGS) -c coa.c -o coa.o

coa: coa.o notorch.o loragrad.o
	$(CC) coa.o notorch.o loragrad.o $(LDFLAGS) -o coa

run: coa
	./coa origin.txt

clean:
	rm -f coa coa.o notorch.o loragrad.o

.PHONY: all run clean
