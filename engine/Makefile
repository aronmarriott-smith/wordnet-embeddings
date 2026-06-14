# Builds the inference engine as a shared library (for ctypes) and a
# C-level test binary.
#
# See CUSTOM_EMBEDDINGS_RESEARCH.md, Part 4 — this mirrors the native-build
# approach already validated for sqlite-vec on the Raspberry Pi 2B
# (PI2B_FEASIBILITY.md): no cross-compilation, build with the system gcc.

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC -Iinclude

SRC = src/embed.c
TEST_SRC = tests/test_engine.c

.PHONY: all lib test clean

all: lib

lib: libembed.so

libembed.so: $(SRC) include/embed.h
	$(CC) $(CFLAGS) -shared -o $@ $(SRC)

test: test_engine
	./test_engine

test_engine: $(TEST_SRC) $(SRC) include/embed.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(SRC)

clean:
	rm -f libembed.so test_engine
