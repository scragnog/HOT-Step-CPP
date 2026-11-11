CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
INCLUDES  = -I../src

test-philox: test-philox.cpp ../src/philox.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -lm

test-ace-parity: test-ace-parity.cpp ../src/request.cpp ../vendor/yyjson/yyjson.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I../vendor/yyjson -o $@ $^ -lm

clean:
	rm -f test-philox test-ace-parity philox-noise.f32

.PHONY: clean
