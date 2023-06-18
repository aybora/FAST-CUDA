#include "fast.cpp"

/*
mode 0: OpenCV FAST
mode 1: OpenCV CUDA FAST
mode 2: FAST CPU written by aybora
mode 3: FAST CUDA (global memory) written by aybora
mode 4: FAST CUDA (shared memory) written by aybora
*/

int main(int argc, char **argv)
{

	parse_args(argc, argv);

    cv::Mat image;
    image = cv::imread(filename, 1);
    const char* outsuffix;
    int circSize = 5;
    clock_t start = clock();
    switch(mode) {
        case 0:
            {
                printf("Running OpenCV FAST on CPU \n");
                FAST_OpenCV_CPU(image, threshold, circSize);
                outsuffix = "_fast_cv";
                break;
            }
        case 1:
            {
                printf("Running OpenCV FAST CUDA on GPU \n");
                FAST_OpenCV_CUDA(image, threshold, circSize);
                outsuffix = "_fast_cv_gpu";
                break;
            }
        case 2:
            {
                printf("Running Naive FAST on CPU \n");
                FAST_CPU(image, threshold, circSize);
                outsuffix = "_fast_cpu";
                break;
            }
        case 3:
            {
                printf("Running FAST CUDA on GPU with global memory \n");
                FAST_GPU(image, threshold, circSize);
                outsuffix = "_fast_gpu";
                break;
            }
        case 4:
            {
                printf("Running FAST CUDA on GPU with shared memory \n");
                FAST_GPU_shared(image, threshold, circSize);
                outsuffix = "_fast_gpu_shared";
                break;
            }
        case 5:
            {
                printf("Running FAST CUDA on GPU with optimized settings \n");
                FAST_GPU_opt(image, threshold, circSize);
                outsuffix = "_fast_gpu_opt";
                break;
            }
    }
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
   
    // Split the filename
    const char* outname = extendFilename(filename, outsuffix);
    cv::imwrite(outname, image);
    printf("Time elapsed for FAST on %i sized image: %f ms\n", image.cols*image.rows, elapsed_time_ms);

}
