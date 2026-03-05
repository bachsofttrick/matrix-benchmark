CXX      = g++
CXXFLAGS = -O2 -mavx2 -mfma -std=c++17
LIBS     = -lvulkan -lpthread

TARGET   = test
SRC      = test.cpp
SHADER   = matmul.comp
SPV      = matmul.spv

.PHONY: all clean

all: $(SPV) $(TARGET)

$(SPV): $(SHADER)
	glslangValidator -V $(SHADER) -o $(SPV)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) $(SPV)
