# Compiler and Language Standard
CXX      := g++-14
CXXFLAGS := -std=c++23 -O3 -Wall -Wextra

# Path to Eigen headers (Update this if Eigen is elsewhere, e.g., /usr/local/include/eigen3)
# On many Linux systems, 'pkg-config' can find this for you automatically.
EIGEN_INC := $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

# Include directories
INCLUDES := -I../include $(EIGEN_INC)

# Target executable name
TARGET   := eigen_test

# Source and Object files
SRCS     := vectors.cpp
OBJS     := $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Linking the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Compiling source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean