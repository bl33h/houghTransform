all: pgm.o hough constantVersion sharedVersion

hough: versions/hough.cu pgm.o
	nvcc versions/hough.cu pgm.o -o hough -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

constantVersion: versions/constant.cu pgm.o
	nvcc versions/constant.cu pgm.o -o constantVersion -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

sharedVersion: versions/shared.cu pgm.o
	nvcc versions/shared.cu pgm.o -o sharedVersion -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

pgm.o: pgm/pgm.cpp
	g++ -c pgm/pgm.cpp -o ./pgm.o -Wno-overloaded-virtual