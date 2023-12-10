#include <iostream>
#include <chrono>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "sgm.h"

#include "sample_common.h"

#include "utility.h"

static const std::string keys =
"{ @left-image-format  | <none> | (required) format string for path to input left image  }"
"{ @right-image-format | <none> | (required) format string for path to input right image }"
"{ @image-format | png | (required) extension of the image file e.g png, jpg etc }"
"{ disparity_path | <none> | (required) Path to store the Gray-Scale disparities }"
"{ colour_disp_path | <none> | (required) Path to store the Colored-Graded disparities  }"
"{ rdleft_path | <none> | (required) Path to store the reduced Left path  }"
"{ disp_size           |    64 | (optional) maximum possible disparity value            }"
"{ help h              |        | (optional) display this help and exit                  }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	// Assess the minimum number of arguments to run the executable
	if(argc < 3)
	{
        std::cout << "Not enough CML parameters" << std::endl;
        parser.printMessage();
		return -1;
	}
    // Print Help Message
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
    // Check if the CMDL has the mandatory keywords for running the executable
	if(!parser.has("@left-image-format") && !parser.has("@right-image-format") && !parser.has("@image-format") && !parser.has("disparity_path") && !parser.has("colour_disp_path") && !parser.has("rdleft_path"))
	{
		std::cout << "" <<std::endl;
		std::cout << "-@left-image-format , -@right-image-format , @image-format , -disparity_path ,  -colour_disp_path and -rdleft_path keywords are required for running the executable!" <<std::endl;
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
    // Define the Variables from Args
	std::string image_format_L = parser.get<cv::String>("@left-image-format");
	std::string image_format_R = parser.get<cv::String>("@right-image-format");
	const std::string image_format = parser.get<cv::String>("@image-format");
	const int disp_size = parser.get<int>("disp_size");
	std::string disp_path = parser.get<cv::String>("disparity_path");
	std::string cdisp_path = parser.get<cv::String>("colour_disp_path");
    std::string rdleft_path = parser.get<cv::String>("rdleft_path");
    // Check the syntax of the Parser if is correct
	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}
	// Check the validity of the paths 
	int flag = 0;
    checkPath(flag,image_format_L,image_format_L);
	if(flag == -1){
		return -1;
	}
	flag = 0;
    checkPath(flag,image_format_R,image_format_R);
	if(flag == -1){
		return -1;
	}
	checkPath(flag,disp_path,disp_path);
	if(flag == -1){
		return -1;
	}
	flag = 0;
    checkPath(flag,cdisp_path,cdisp_path);
	if(flag == -1){
		return -1;
	}
	flag = 0;
    checkPath(flag,rdleft_path,rdleft_path);
	if(flag == -1){
		return -1;
	}
    // Load the left and right image
	std::vector<cv::String> l_file_name, r_file_name;
	// check if the image format is correct
	// Insert your path to the image folder in the " " of glob
	cv::glob(image_format_L+"*."+image_format,l_file_name,false);
	cv::glob(image_format_R+"*."+image_format,r_file_name,false);
	std::vector<cv::Mat> l_images, r_images, disparities, colorize_disparities, lefts_red;
	size_t l_count = l_file_name.size();
    size_t r_count = r_file_name.size();
	std::cout << "Left Images Numbers :  " << l_count << std::endl;
	std::cout << "Right Images Numbers : " << r_count  << std::endl;
    std::cout << "0 - Pre Load Images " << std::endl;
	flag = retrieveImage(l_count,l_file_name,l_images);
	if(flag == -1){
		return -1;
	}
	flag = retrieveImage(r_count,r_file_name,r_images);
	if(flag == -1){
		return -1;
	}
    std::cout << "1 - Compute the Disparities and Post Processing " << std::endl;
	//double scale_down = 0.4;
    int down_width = 300;
    int down_height = 200;
	for (size_t i = 0;i < l_images.size(); i++){
        // Pick the current set of images
		cv::Mat I1 = l_images[i];
		cv::Mat I2 = r_images[i];

        // Modified here
		resize(I1,I1, cv::Size(down_width, down_height), cv::INTER_LINEAR);
		resize(I2,I2, cv::Size(down_width, down_height), cv::INTER_LINEAR);


        // Define the SGM Parameters
		const int width = I1.cols;
	    const int height = I1.rows;
	    const int src_depth = I1.type() == CV_8U ? 8 : 16;
	    const int dst_depth = disp_size < 256 ? 8 : 16;
	    const int src_bytes = src_depth * width * height / 8;
	    const int dst_bytes = dst_depth * width * height / 8;
        // Define the Celsus SGM
	    sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
		device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
	    cv::Mat disparity(height, width, dst_depth == 8 ? CV_8S : CV_16S), disparity_color;
        // Obtain the Raw Disparity
	    const int invalid_disp = sgm.get_invalid_disparity();
        // Upload the images to the GPU
		d_I1.upload(I1.data);
		d_I2.upload(I2.data);
        // Initialize the timer
		const auto t1 = std::chrono::system_clock::now();
        // Execute Celsus SGM
		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		// Counting the timing of the approach
		cudaDeviceSynchronize();
		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;
        // Store the images to the CPU
		d_disparity.download(disparity.data);
		// Normalize the left image if is not in the INT8U format
		if (I1.type() != CV_8U){
		    std::cout << "Normalized! " << std::endl;
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX, CV_8U);
		}
        // Create the Color Grid of the Disparity
		colorize_disparity(disparity, disparity_color, disp_size, disparity == invalid_disp);
		std::cout << "SGM Execution Time : " << 1e-3 * duration << " [msec] " << fps << "  [FPS]" << std::endl;
		// Downscale the Left Image, the Disparity and the Colored Disparity by 0.4
		//resize(I1,I1, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR); 
		//resize(I1,I1, cv::Size(down_width, down_height), cv::INTER_LINEAR); 
		disparity.convertTo(disparity, CV_8U);
        //resize(disparity,disparity, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR);  
		//resize(disparity,disparity, cv::Size(down_width, down_height), cv::INTER_LINEAR);   
		disparity_color.convertTo(disparity_color, CV_32F);
        //resize(disparity_color,disparity_color, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR); 
		//resize(disparity_color,disparity_color, cv::Size(down_width, down_height), cv::INTER_LINEAR); 
		// Store the results in a vector of images
		lefts_red.push_back(I1);
        disparities.push_back(disparity);
		colorize_disparities.push_back(disparity_color);
	}
	// Save the images to a Path
	std::cout << "2 - Store the Disparities and the Reduced Left Image " << std::endl;
	saveImg(lefts_red, rdleft_path);
	saveImg(disparities, disp_path);
	saveImg(colorize_disparities, cdisp_path);
	return 0;
}
