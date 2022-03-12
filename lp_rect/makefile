CXX = g++
CXXFLAGS = -std=c++11  -Wall -g
OPENCV_DIR=/usr/local/include/opencv4/
OPENCV_BUILD_DIR=/usr/local/lib

lp_rect: lp_rect.o 
	$(CXX) $(CXXFLAGS) -v lp_rect.o  \
	-L '$(OPENCV_BUILD_DIR)' \
	-lopencv_photo \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-o lp_rect

lp_rect.o: 
	$(CXX) $(CXXFLAGS) \
	-I$(OPENCV_DIR) \
	-I$(OPENCV_DIR)opencv2/core \
	-I$(OPENCV_DIR)opencv2/imgproc \
	-I$(OPENCV_DIR)opencv2/imgcodecs \
	-Wall -c lp_rect.cpp


clean:
	rm lp_rect lp_rect.o 