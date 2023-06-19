# FAST-CUDA

This work can be compiled with the following command on an OpenCV CUDA installed computer:

nvcc main.cu -lopencv\_core -lopencv\_highgui -lopencv\_imgproc -lopencv\_imgcodecs -lopencv\_features2d -lopencv\_cudafeatures2d -o ./fast

Then, code can be run with the following command:

./test -f file -t th -m mode

Where file corresponds to filename, th corresponds to the threshold value (0-255) and mode corresponds to the operation mode. 
Modes: 0: OpenCV, 1: OpenCV GPU, 2: Naive CPU, 3: GPU global, 4: GPU shared, 5: GPU optimized.
