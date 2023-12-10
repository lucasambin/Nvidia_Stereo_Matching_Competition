#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <stdlib.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "disparity_method.h"
#include "utility.h"

static const std::string keys =
"{ @left-image-format  | <none> | (required) format string for path to input left image  }"
"{ @right-image-format | <none> | (required) format string for path to input right image }"
"{ disparity_path      | <none> | (required) Path to store the PFM file }"
"{ p1              |   true | (required) SGM parameter 1 }"
"{ p2              |   true | (required) SGM parameter 2 }"
"{ help h              |        | display this help and exit                  }";

int main(int argc, char *argv[]) {
        cv::CommandLineParser parser(argc, argv, keys);
        if (parser.has("help")) {
		parser.printMessage();
		return 0;
        }
	if(argc < 4) {
		std::cerr << "Not enough CML parameters! Usage: cuda_sgm dir p1 p2" << std::endl;
                parser.printMessage();
		return -1;
	}
	if(MAX_DISPARITY != 128) {
		std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
		return -1;
	}
	if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
                std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
                return -1;
        }
        // Check if the CMDL has the mandatory keywords for running the executable
        if(!parser.has("@left-image-format") && !parser.has("@right-image-format") && !parser.has("disparity_path") && !parser.has("p1") && !parser.has("p2"))
        {
		std::cout << "" <<std::endl;
		std::cout << "-@left-image-format , -@right-image-format , disparity_path, p1 and p2 keywords are required for running the executable!" <<std::endl;
		std::cout << "" <<std::endl;
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
                std::cout << "" <<std::endl;
		parser.printMessage();
		return -1;
        }
        // Check the CMDL syntax if is correct
        if(!parser.check())
        {
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
                parser.printMessage();
		return -1;
        }
        std::string image_format_L = parser.get<cv::String>("@left-image-format");
        std::string image_format_R = parser.get<cv::String>("@right-image-format");
        std::string disp_path = parser.get<cv::String>("disparity_path");
        int p1 = parser.get<int>("p1");
        int p2 = parser.get<int>("p2");
	std::vector<float> times;
        bool resize = false;
        // Size of Kitti Dataset to be used for converting every type of image due to Algorithm constraints : 1280×384
        int kitti_width = 1280;
        int kitti_height = 384;
        // Initialize the SGM parameters P1 and P2
	init_disparity_method(p1, p2);
        // Read the images
        cv::Mat h_im0 = cv::imread(image_format_L, cv::IMREAD_GRAYSCALE );
        cv::Mat h_im1 = cv::imread(image_format_R, cv::IMREAD_GRAYSCALE );
        cv::Size output_size = h_im0.size();
        // Check if the images are loaded and in the correct size
        if(!h_im0.data) {
            std::cerr << "Couldn't read the file " << image_format_L << std::endl;
            return EXIT_FAILURE;
	}
	if(!h_im1.data) {
            std::cerr << "Couldn't read the file " << image_format_R << std::endl;
	    return EXIT_FAILURE;
	}
        if(h_im0.rows != h_im1.rows || h_im0.cols != h_im1.cols) {
	    std::cerr << "Both images must have the same dimensions" << std::endl;
	    return EXIT_FAILURE;
	}
        // If the image does not have size as Kitti dataset then resize them as the same size of a Kitti's image 
	if(h_im0.rows % 4 != 0 || h_im0.cols % 4 != 0) {
            resize = true;
            cv::resize(h_im0,h_im0, cv::Size(kitti_width, kitti_height), cv::INTER_LINEAR);
            cv::resize(h_im1,h_im1, cv::Size(kitti_width, kitti_height), cv::INTER_LINEAR);
	}

#if LOG
       std::cout << "processing: " << image_format_L << std::endl;
#endif
       // Execute SGM 2016 for the given pair of images
       std::cout << "1 - Compute the Disparities and Post Processing " << std::endl;
       float elapsed_time_ms;
       cv::Mat disparity_im = compute_disparity_method(h_im0, h_im1, &elapsed_time_ms);
#if LOG
       std::cout << "done" << std::endl;
#endif
       times.push_back(elapsed_time_ms);
#if WRITE_FILES
       std::cout << "2 - Write the output as PFM file " << std::endl;
       // Convert the Disparity to 32FC1 and scaling
       disparity_im.convertTo(disparity_im, CV_32FC1, 1.0);
       if(resize)
       {
          /* In order to retrive the correct disparity due the downscaling we 
                  1 - Resize with Nearest Neighbour Interlinear,
                  2 - Retrieve the Scale Factor = original size width / (float) Kitti size width
                  3 - Then scale the disparity as Disparity Map * Scale Factor
          */
          cv::resize(disparity_im,disparity_im, output_size, cv::INTER_NEAREST);
          float res_scale_w = output_size.width/(float)kitti_width;
          float res_scale = res_scale_w;
          disparity_im *= res_scale;
       }
       // Store the Disparity as a .pfm file
       savePFM(disparity_im, disp_path);
	
#endif
      // Erase all the variables stored in the GPU
      finish_disparity_method();
      // Obtain the overall SGM 2016 execution and print to the terminal 
      double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
      std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
      return 0;
}
