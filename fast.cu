#include "fast.hpp"

struct CornerPoint
{
    int x;
    int y;
};

void FAST_OpenCV_CUDA(cv::Mat image, int threshold, int circSize) {
    
    std::vector<cv::KeyPoint> key;
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat imageGpu;
    imageGpu.upload(imageGray);
    cv::Ptr<cv::cuda::FastFeatureDetector> detector = cv::cuda::FastFeatureDetector::create(threshold, true);
    clock_t start = clock();
    detector->detect(imageGpu, key, cv::cuda::GpuMat());
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);
    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

}

// Function to calculate the score for a corner based on the pixel value differences
__device__ __host__ int calculateScore(const uchar* image, int width, int x, int y, int threshold)
{
    const int offsets[16][2] = {
        { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 },
        { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }
    };

    int score = 0;
    for (int i = 0; i < 16; i++)
    {
        int xx = x + offsets[i][0];
        int yy = y + offsets[i][1];
        if (image[yy * width + xx] - image[y * width + x] > threshold) // Brighter
            score += image[yy * width + xx] - image[y * width + x];
        else if (image[y * width + x] - image[yy * width + xx] > threshold) // Darker
            score += image[yy * width + xx] - image[y * width + x];
    }
    return score;
}

__global__ void calculateScoresAndMaximas(
    const uchar* image, int width, int height, int threshold, 
    CornerPoint* corners, int cornersCount, 
    bool* maximas, CornerPoint* cornersNMS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < cornersCount) {
        int x = corners[i].x;
        int y = corners[i].y;

        int score = calculateScore(image, width, x, y, threshold);
        bool is_maxima = true;
        
        for (int j = 0; j < cornersCount; j++) {
            float dx = corners[j].x - x;
            float dy = corners[j].y - y;

            if (dx * dx + dy * dy <= 3 * 3) { // Check distance within 3 pixels
                int existing_score = calculateScore(image, width, corners[j].x, corners[j].y, threshold);

                if (std::abs(score) < std::abs(existing_score)) {
                    is_maxima = false;
                    break;
                }
            }
        }

        maximas[i] = is_maxima;
        cornersNMS[i].x = x;
        cornersNMS[i].y = y;
    }
}

__global__ void calculateScoresAndMaximas_shared(
    const uchar* image, int width, int height, int threshold, 
    CornerPoint* corners, int cornersCount, 
    bool* maximas, CornerPoint* cornersNMS)
{
    // Calculate the global index for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Create shared memory for this block's corners
    extern __shared__ CornerPoint sharedCorners[];

    // Each thread loads one corner into shared memory
    if(i < cornersCount) {
        sharedCorners[threadIdx.x] = corners[i];
    }

    // Ensure all threads have finished loading before proceeding
    __syncthreads();

    // Only continue if this thread's global index is within bounds
    if(i < cornersCount) {
        int x = sharedCorners[threadIdx.x].x;
        int y = sharedCorners[threadIdx.x].y;

        int score = calculateScore(image, width, x, y, threshold);
        bool is_maxima = true;

        // Only loop over corners in shared memory
        for (int j = 0; j < blockDim.x; j++) {
            float dx = sharedCorners[j].x - x;
            float dy = sharedCorners[j].y - y;

            if (dx * dx + dy * dy <= 3 * 3) { // Check distance within 3 pixels
                int existing_score = calculateScore(image, width, sharedCorners[j].x, sharedCorners[j].y, threshold);

                if (std::abs(score) < std::abs(existing_score)) {
                    is_maxima = false;
                    break;
                }
            }
        }

        maximas[i] = is_maxima;
        cornersNMS[i].x = x;
        cornersNMS[i].y = y;
    }
}

// Function to check if a pixel is brighter or darker than the center pixel
__device__ __host__ bool isCorner(const uchar* image, int width, int x, int y, int threshold)
{
    int center_value = image[y * width + x];
    int brighter_count = 0;
    int darker_count = 0;

    // Check the four quadrants
    if (image[(y - 3) * width + x] - center_value > threshold) // Top
        brighter_count++;
    else if (center_value - image[(y - 3) * width + x] > threshold)
        darker_count++;

    if (image[y * width + (x + 3)] - center_value > threshold) // Right
        brighter_count++;
    else if (center_value - image[y * width + (x + 3)] > threshold)
        darker_count++;

    if (image[(y + 3) * width + x] - center_value > threshold) // Bottom
        brighter_count++;
    else if (center_value - image[(y + 3) * width + x] > threshold)
        darker_count++;

    if (image[y * width + (x - 3)] - center_value > threshold) // Left
        brighter_count++;
    else if (center_value - image[y * width + (x - 3)] > threshold)
        darker_count++;

    return (brighter_count >= 1 || darker_count >= 1);
}

__global__ void detectCornersCUDA(const uchar* image, int width, int height, int threshold, CornerPoint* corners, int* cornersCount)
{
    const int offsets[16][2] = {
        { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 },
        { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }
    };

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= 3 && x < width - 3 && y >= 3 && y < height - 3)
    {
        if (isCorner(image, width, x, y, threshold))
        {
            int brighter_count = 0;
            int darker_count = 0;
            for (int i = 0; i < 25; i++)
            {
                int xx = x + offsets[i%16][0];
                int yy = y + offsets[i%16][1];
                if (image[yy * width + xx] - image[y * width + x] > threshold) // Brighter
                    {
                    brighter_count++;
                    darker_count=0;
                    }
                else if (image[y * width + x] - image[yy * width + xx] > threshold) // Darker
                    {
                    darker_count++;
                    brighter_count=0;
                    }
                else 
                    {
                    darker_count=0;
                    brighter_count=0;
                    }
                if (brighter_count >= 9 || darker_count >= 9) 
                
                    {
                        int index = atomicAdd(cornersCount, 1);
                        corners[index].x = x;
                        corners[index].y = y;
                        break;
                    }
                }
            }
    }
}

__global__ void detectCornersCUDA_shared(const uchar* image, int width, int height, int threshold, CornerPoint* corners, int* cornersCount)
{
    const int offsets[16][2] = {
        { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 },
        { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }
    };

    // Create shared memory for the image block.
    __shared__ uchar shared_image[BLOCK_SIZE + 6][BLOCK_SIZE + 6];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Load image data into shared memory.
    if (x < width && y < height)
    {
        shared_image[threadIdx.y + 3][threadIdx.x + 3] = image[y * width + x];
    }

    // Load halo elements into shared memory.
    if (threadIdx.x < 3)
    {
        if (x >= 3)
        {
            shared_image[threadIdx.y + 3][threadIdx.x] = image[y * width + (x - 3)];
        }
        if (x < width - 3)
        {
            shared_image[threadIdx.y + 3][threadIdx.x + 3 + BLOCK_SIZE] = image[y * width + (x + BLOCK_SIZE)];
        }
    }

    if (threadIdx.y < 3)
    {
        if (y >= 3)
        {
            shared_image[threadIdx.y][threadIdx.x + 3] = image[(y - 3) * width + x];
        }
        if (y < height - 3)
        {
            shared_image[threadIdx.y + 3 + BLOCK_SIZE][threadIdx.x + 3] = image[(y + BLOCK_SIZE) * width + x];
        }
    }

    // Ensure all the loading into shared memory is completed.
    __syncthreads();

    if (x >= 3 && x < width - 3 && y >= 3 && y < height - 3)
    {
        if (isCorner(&shared_image[0][0], BLOCK_SIZE + 6, threadIdx.x + 3, threadIdx.y + 3, threshold))
        {
            int brighter_count = 0;
            int darker_count = 0;
            for (int i = 0; i < 25; i++)
            {
                int xx = threadIdx.x + 3 + offsets[i%16][0];
                int yy = threadIdx.y + 3 + offsets[i%16][1];
                if (shared_image[yy][xx] - shared_image[threadIdx.y + 3][threadIdx.x + 3] > threshold) // Brighter
                {
                    brighter_count++;
                    darker_count=0;
                }
                else if (shared_image[threadIdx.y + 3][threadIdx.x + 3] - shared_image[yy][xx] > threshold) // Darker
                {
                    darker_count++;
                    brighter_count=0;
                }
                else 
                {
                    darker_count=0;
                    brighter_count=0;
                }
                if (brighter_count >= 9 || darker_count >= 9) 
                {
                    int index = atomicAdd(cornersCount, 1);
                    corners[index].x = x;
                    corners[index].y = y;
                    break;
                }
            }
        }
    }
}

void FAST_GPU(cv::Mat image, int threshold, int circSize) {
    
    //Convert image to grayscale
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    //Save image size info
    int width = imageGray.cols;
    int height = imageGray.rows;
    int imageSize = width * height * sizeof(uchar);

    //Allocate memories for host and device
    uchar* h_image;
    h_image = (uchar *)malloc(imageSize);
    memcpy(h_image, imageGray.data, imageSize);

    uchar* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    //Copy image data to device
    cudaMemcpy(d_image, imageGray.data, imageSize, cudaMemcpyHostToDevice);

    //Allocate device memory for corner info
    CornerPoint* d_corners;
    cudaMalloc((void**)&d_corners, width * height * sizeof(CornerPoint));
    int* d_cornersCount;
    cudaMalloc((void**)&d_cornersCount, sizeof(int));
    cudaMemset(d_cornersCount, 0, sizeof(int));

    //Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //Run kernel
    
    detectCornersCUDA<<<grid, block>>>(d_image, width, height, threshold, d_corners, d_cornersCount);
    
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_1;
    
    cudaEventElapsedTime(&elapsed_1, start, stop);

    int h_cornersCount;
    cudaMemcpy(&h_cornersCount, d_cornersCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    bool* d_maximas;
    CornerPoint* d_cornersNMS;

    cudaMalloc(&d_maximas, h_cornersCount * sizeof(bool));
    cudaMalloc(&d_cornersNMS, h_cornersCount * sizeof(CornerPoint));

    // Run kernel
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    int blocksPerGrid = (h_cornersCount + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start, 0);

    calculateScoresAndMaximas<<<blocksPerGrid, threadsPerBlock>>>(
        d_image, width, height, threshold, d_corners, h_cornersCount, d_maximas, d_cornersNMS);
    
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_2;
    
    cudaEventElapsedTime(&elapsed_2, start, stop);

    // Copy results back to CPU
    bool* h_maximas = new bool[h_cornersCount];
    CornerPoint* h_cornersNMS = new CornerPoint[h_cornersCount];
    cudaMemcpy(h_maximas, d_maximas, h_cornersCount * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cornersNMS, d_cornersNMS, h_cornersCount * sizeof(CornerPoint), cudaMemcpyDeviceToHost);

    // Check for maxima and add to result
    std::vector<cv::KeyPoint> key;
    for(int i = 0; i < h_cornersCount; i++) {
        if(h_maximas[i]) {
            int x = h_cornersNMS[i].x;
            int y = h_cornersNMS[i].y;
            key.push_back(cv::KeyPoint(x, y, 7));
        }
    }
    
    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

    float elapsed_time_ms = elapsed_1 + elapsed_2;
    
    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);

    cudaFree(d_image);
    cudaFree(d_corners);
    cudaFree(d_cornersCount);
    cudaFree(d_cornersNMS);
    cudaFree(d_maximas);
    delete[] h_cornersNMS;
    delete[] h_maximas;
    free(h_image);

}

void FAST_GPU_shared(cv::Mat image, int threshold, int circSize) {

    //Convert image to grayscale
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    //Save image size info
    int width = imageGray.cols;
    int height = imageGray.rows;
    int imageSize = width * height * sizeof(uchar);

    //Allocate memories for host and device
    uchar* h_image;
    h_image = (uchar *)malloc(imageSize);
    memcpy(h_image, imageGray.data, imageSize);

    uchar* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    //Copy image data to device
    cudaMemcpy(d_image, imageGray.data, imageSize, cudaMemcpyHostToDevice);

    //Allocate device memory for corner info
    CornerPoint* d_corners;
    cudaMalloc((void**)&d_corners, width * height * sizeof(CornerPoint));
    int* d_cornersCount;
    cudaMalloc((void**)&d_cornersCount, sizeof(int));
    cudaMemset(d_cornersCount, 0, sizeof(int));

    //Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Calculate shared memory size in bytes
    int sharedMemSize = (BLOCK_SIZE + 6) * (BLOCK_SIZE + 6) * sizeof(uchar);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //Run kernel
    detectCornersCUDA_shared<<<grid, block, sharedMemSize>>>(d_image, width, height, threshold, d_corners, d_cornersCount);

    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_1;
    
    cudaEventElapsedTime(&elapsed_1, start, stop);
    
    int h_cornersCount;
    cudaMemcpy(&h_cornersCount, d_cornersCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    bool* d_maximas;
    CornerPoint* d_cornersNMS;

    cudaMalloc(&d_maximas, h_cornersCount * sizeof(bool));
    cudaMalloc(&d_cornersNMS, h_cornersCount * sizeof(CornerPoint));

    // Run kernel
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    int blocksPerGrid = (h_cornersCount + threadsPerBlock - 1) / threadsPerBlock;
    // Calculate shared memory size in bytes
    sharedMemSize = threadsPerBlock * sizeof(CornerPoint);

    cudaEventRecord(start, 0);

    calculateScoresAndMaximas_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_image, width, height, threshold, d_corners, h_cornersCount, d_maximas, d_cornersNMS);
    
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_2;
    
    cudaEventElapsedTime(&elapsed_2, start, stop);

    // Copy results back to CPU
    bool* h_maximas = new bool[h_cornersCount];
    CornerPoint* h_cornersNMS = new CornerPoint[h_cornersCount];
    cudaMemcpy(h_maximas, d_maximas, h_cornersCount * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cornersNMS, d_cornersNMS, h_cornersCount * sizeof(CornerPoint), cudaMemcpyDeviceToHost);

    // Check for maxima and add to result
    std::vector<cv::KeyPoint> key;
    for(int i = 0; i < h_cornersCount; i++) {
        if(h_maximas[i]) {
            int x = h_cornersNMS[i].x;
            int y = h_cornersNMS[i].y;
            key.push_back(cv::KeyPoint(x, y, 7));
        }
    }
    
    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

    float elapsed_time_ms = elapsed_1 + elapsed_2;

    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);

    cudaFree(d_image);
    cudaFree(d_corners);
    cudaFree(d_cornersCount);
    cudaFree(d_cornersNMS);
    cudaFree(d_maximas);
    delete[] h_cornersNMS;
    delete[] h_maximas;
    free(h_image);

}

void FAST_GPU_opt(cv::Mat image, int threshold, int circSize) {

    //Convert image to grayscale
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    //Save image size info
    int width = imageGray.cols;
    int height = imageGray.rows;
    int imageSize = width * height * sizeof(uchar);

    //Allocate memories for host and device
    uchar* h_image;
    h_image = (uchar *)malloc(imageSize);
    memcpy(h_image, imageGray.data, imageSize);

    uchar* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    //Copy image data to device
    cudaMemcpy(d_image, imageGray.data, imageSize, cudaMemcpyHostToDevice);

    //Allocate device memory for corner info
    CornerPoint* d_corners;
    cudaMalloc((void**)&d_corners, width * height * sizeof(CornerPoint));
    int* d_cornersCount;
    cudaMalloc((void**)&d_cornersCount, sizeof(int));
    cudaMemset(d_cornersCount, 0, sizeof(int));

    //Define block and grid sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //Run kernel
    detectCornersCUDA<<<grid, block>>>(d_image, width, height, threshold, d_corners, d_cornersCount);
    
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_1;
    
    cudaEventElapsedTime(&elapsed_1, start, stop);
    
    int h_cornersCount;
    cudaMemcpy(&h_cornersCount, d_cornersCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    bool* d_maximas;
    CornerPoint* d_cornersNMS;

    cudaMalloc(&d_maximas, h_cornersCount * sizeof(bool));
    cudaMalloc(&d_cornersNMS, h_cornersCount * sizeof(CornerPoint));

    // Run kernel
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    int blocksPerGrid = (h_cornersCount + threadsPerBlock - 1) / threadsPerBlock;
    // Calculate shared memory size in bytes
    int sharedMemSize = threadsPerBlock * sizeof(CornerPoint);

    cudaEventRecord(start, 0);

    calculateScoresAndMaximas_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_image, width, height, threshold, d_corners, h_cornersCount, d_maximas, d_cornersNMS);
    
    cudaEventRecord(stop, 0);

    cudaDeviceSynchronize();

    float elapsed_2;
    
    cudaEventElapsedTime(&elapsed_2, start, stop);

    // Copy results back to CPU
    bool* h_maximas = new bool[h_cornersCount];
    CornerPoint* h_cornersNMS = new CornerPoint[h_cornersCount];
    cudaMemcpy(h_maximas, d_maximas, h_cornersCount * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cornersNMS, d_cornersNMS, h_cornersCount * sizeof(CornerPoint), cudaMemcpyDeviceToHost);

    // Check for maxima and add to result
    std::vector<cv::KeyPoint> key;
    for(int i = 0; i < h_cornersCount; i++) {
        if(h_maximas[i]) {
            int x = h_cornersNMS[i].x;
            int y = h_cornersNMS[i].y;
            key.push_back(cv::KeyPoint(x, y, 7));
        }
    }
    
    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

    float elapsed_time_ms = elapsed_1 + elapsed_2;

    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);

    cudaFree(d_image);
    cudaFree(d_corners);
    cudaFree(d_cornersCount);
    cudaFree(d_cornersNMS);
    cudaFree(d_maximas);
    delete[] h_cornersNMS;
    delete[] h_maximas;
    free(h_image);

}