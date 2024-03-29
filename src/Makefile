CC = clang
CXXFLAGS = -std=c++17 -Wall
INCLUDE_PATH = -I/usr/local/neuware/include
LIBS = -lstdc++

all: mlu_test

mlu_test: mlu_comm.cc tool.cc
	$(CC) $(CXXFLAGS) $(INCLUDE_PATH) $^ -o $@ $(LIBS)

clean:
	rm -f mlu_test
