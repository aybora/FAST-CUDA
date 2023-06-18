#include "fast.cu"

void FAST_OpenCV_CPU(cv::Mat image, int threshold, int circSize) {
    
    std::vector<cv::KeyPoint> key;
    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, true);
    clock_t start = clock();
    detector->detect(imageGray, key, cv::Mat());
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);
    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

}

// Function to detect corners using the FAST algorithm
std::vector<cv::KeyPoint> detectCorners(const uchar* image, int width, int height, int threshold)
{
    const int offsets[16][2] = {
        { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 },
        { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }
    };

    std::vector<cv::KeyPoint> corners;
    for (int y = 3; y < height - 3; y++)
    {
        for (int x = 3; x < width - 3; x++)
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
                            corners.push_back(cv::KeyPoint(x, y, 7));
                        }
                }
            }
        }
    }
    return corners;
}

__host__ std::vector<cv::KeyPoint> NMS(const uchar* image, int width, int height, int threshold, CornerPoint* corners, int cornersCount)
{
    std::vector<cv::KeyPoint> cornersNMS;

    for (int i = 0; i < cornersCount; i++)
        {
            int x = corners[i].x;
            int y = corners[i].y;

            int score = calculateScore(image, width, x, y, threshold);
            bool is_maxima = true;
            for (int j = 0; j < cornersCount; j++)
                {
                    float dx = corners[j].x - x;
                    float dy = corners[j].y - y;
                    if (dx * dx + dy * dy <= 3 * 3) // Check distance within 3 pixels
                    {
                        int existing_score = calculateScore(image, width, static_cast<int>(corners[j].x), static_cast<int>(corners[j].y), threshold);
                        if (std::abs(score) < std::abs(existing_score))
                        {
                            is_maxima = false;
                            break;
                        }
                    }
                }
            if (is_maxima)
                {
                    cornersNMS.push_back(cv::KeyPoint(x, y, 7));
                }

        
        }   

    return cornersNMS;
    
}

void FAST_CPU(cv::Mat image, int threshold, int circSize) {

    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    int width = image.cols;
    int height = image.rows;
    int imageSize = width * height * sizeof(uchar);

    uchar* h_image;
    h_image = (uchar *)malloc(imageSize);
    memcpy(h_image, imageGray.data, imageSize);

    clock_t start_one = clock();
    std::vector<cv::KeyPoint> corners = detectCorners(h_image, width, height, threshold);
    clock_t end_one = clock();

    CornerPoint* h_corners = new CornerPoint[width * height];
    int h_cornersCount = corners.size();
    for (int i = 0; i < corners.size(); i++) {
        h_corners[i].x = corners[i].pt.x;
        h_corners[i].y = corners[i].pt.y;
    }
    
    clock_t start_two = clock();
    std::vector<cv::KeyPoint> key = NMS(h_image, width, height, threshold, h_corners, h_cornersCount);
    clock_t end_two = clock();

    double elapsed_time_ms = 1000 * ((double)(end_one - start_one) + (double)(end_two - start_two)) / CLOCKS_PER_SEC;

    printf("Time elapsed for running kernels: %f ms \n" , elapsed_time_ms);

    for (int i = 0; i < key.size(); i++) {
        cv::circle(image, key[i].pt, circSize, cv::Scalar(0, 255, 0), 2);
    }

}