# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -g

# OpenCV Paths
OPENCV_DIR = /usr/local/include/opencv4/
OPENCV_BUILD_DIR = /usr/local/lib
OPENCV_LIBS = -lopencv_photo -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# Targets
TARGETS = lp_rect lp_rect_clean

all: $(TARGETS)

lp_rect: lp_rect.o
	$(CXX) $(CXXFLAGS) $< -L$(OPENCV_BUILD_DIR) $(OPENCV_LIBS) -o $@

lp_rect_clean: lp_rect_clean.o
	$(CXX) $(CXXFLAGS) $< -L$(OPENCV_BUILD_DIR) $(OPENCV_LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENCV_DIR) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o
