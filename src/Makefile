CXX      := g++-14
CXXFLAGS := -std=c++23 -O3 -Wall -Wextra

EIGEN_INC := $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

INCLUDES := -I../include $(EIGEN_INC)

TARGET   := eigen_test

SRCS     := linear_systems_eigenvalues.cpp
OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean