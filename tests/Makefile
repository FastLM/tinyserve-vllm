CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -I../include -I../host/include
LIB_SRC = \
	../selector/bounding_box.cpp \
	../selector/topk.cpp \
	../attention/sparse_attention.cpp \
	../cache/page_cache.cpp \
	../host/src/api.cpp

.PHONY: all test clean

all: test_selector test_attention

test_selector: test_selector.cpp $(LIB_SRC)
	$(CXX) $(CXXFLAGS) -o $@ test_selector.cpp $(LIB_SRC)

test_attention: test_attention.cpp $(LIB_SRC)
	$(CXX) $(CXXFLAGS) -o $@ test_attention.cpp $(LIB_SRC)

test: all
	./test_selector
	./test_attention
	PYTHONPATH=../host/python python3 -m unittest discover -s python -v

clean:
	rm -f test_selector test_attention
