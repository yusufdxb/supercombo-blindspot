# Makefile for e6_monitor standalone build + benchmark
#
# Usage:
#   make          # build + run the benchmark / agreement check
#   make bench    # same
#   make clean    # remove build artifacts

CXX      ?= g++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra -pedantic

BIN = e6_bench

all: bench

bench: $(BIN)
	./$(BIN)

$(BIN): bench_main.cpp e6_monitor.hpp
	$(CXX) $(CXXFLAGS) -o $(BIN) bench_main.cpp

clean:
	rm -f $(BIN)

.PHONY: all bench clean
