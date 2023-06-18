#include <stdint.h>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <time.h>

#include <iostream>
#include <string>
#include <utility>

#define BLOCK_SIZE 8

/// argument parsing
int threshold = 75;
int mode = 3;
char *filename = NULL;

void parse_args(int argc, char **argv){
	for (size_t i = 1; i < argc; i++)
	{
		std::string arg = std::string(argv[i]);
		if (arg == "-f") filename = argv[i + 1];
		if (arg == "-m") mode = atoi(argv[i + 1]);
		if (arg == "-t") threshold = atoi(argv[i + 1]);
	}
	if (filename == NULL) {
		printf("\n--- Path to image must be specified in arguments ... quiting ---");
		exit(1);
	}
	if (mode < 0 || mode > 5) {
		printf("\n--- Mode must be in range 0 - 5 ... quiting ---");
		exit(1);
	}
	if (threshold < 0 || threshold > 255) {
		printf("\n--- Threshold must be in range 0 - 255 ... quiting ---");
		exit(1);
	}
	printf("\n--- Running with following setup: --- \n");
	printf("     Threshold: %d\n", threshold);
	printf("     Mode: %d\n", mode);
	printf("     File name: %s\n", filename);
	return;
}

const char* extendFilename(const char* filename, const char* outsuffix) {
    // Convert to std::string for easy manipulation
    std::string filename_str(filename);
	std::string outsuffix_str(outsuffix);

    // Find the position of the dot, starting from the end
    size_t dot_pos = filename_str.find_last_of('.');

    // Split the filename and extension
    std::string file_str = filename_str.substr(0, dot_pos);
    std::string ext_str = filename_str.substr(dot_pos);

	std::string outfile_str = file_str + outsuffix_str + ext_str;

    // Return the results as a pair of strings

	// Create a dynamically allocated copy of the string data
    char* outfile = new char[outfile_str.length() + 1];
    std::strcpy(outfile, outfile_str.c_str());

    return outfile;
}