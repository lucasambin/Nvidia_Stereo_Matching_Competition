#include <iostream>
#include <chrono>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "sgm.h"
#include "utility.h"
#include "sample_common.h"

static const std::string keys =
"{ @left-image-format  | <none> | (required) format string for path to input left image  }"
"{ @right-image-format | <none> | (required) format string for path to input right image }"
"{ disparity_path      | <none> | (optional) Path to store the PFM file }"
"{ disp_size           |    64  | (optional) maximum possible disparity value }"
"{ p1                  |   10   | (optional) SGM Penalization parameter P1 < P2}"
"{ p2                  |   120  | (optional) SGM Smoothing parameter P2 > P1 }"
"{ path_type           |   0    | (optional) SGM Path it could be 4 = 0 or 8 = 1 }"
"{ census_type         |   1    | (optional) SGM Census type could be CENSUS_9x7 = 0 or SYMMETRIC_CENSUS_9x7 = 1 }"
"{ sub_p               |  true  | (optional) Sub Pixel precision to enhance the Disparity Map result }"
"{ reduce              |  false | (optional) reduce the size of the image before run Celsus SGM }"
"{ help h              |        | display this help and exit }";

int main(int argc, char* argv[])
{
       cv::CommandLineParser parser(argc, argv, keys);
       if (parser.has("help")) {
		parser.printMessage();
		return 0;
       }
       if(argc < 3)
       {
                std::cout << "Not enough CML parameters" << std::endl;
                parser.printMessage();
		return -1;
       }
       // Check if the CMDL has the mandatory keywords for running the executable
       if(!parser.has("@left-image-format") && !parser.has("@right-image-format") && !parser.has("disparity_path"))
       {
		std::cout << "" <<std::endl;
		std::cout << "-@left-image-format , -@right-image-format and disparity_path keywords are required for running the executable!" <<std::endl;
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
       bool reduce = parser.get<bool>("reduce");
       int disp_size = parser.get<int>("disp_size");    
       if (!parser.check()) 
       {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
       }
       int down_width = 300;
       int down_height = 200;
       // Read the images in GrayScale
       cv::Mat I1 = cv::imread(image_format_L, cv::IMREAD_GRAYSCALE );
       cv::Mat I2 = cv::imread(image_format_R, cv::IMREAD_GRAYSCALE );
       cv::Size output_size = I1.size();
       // Downscale the Left and Right image to fast the computation of the Disparity Map
       if(reduce)
       {
                 cv::resize(I1,I1, cv::Size(down_width, down_height), cv::INTER_LINEAR);
                 cv::resize(I2,I2, cv::Size(down_width, down_height), cv::INTER_LINEAR);
       }
       // Define the SGM Parameters
       const int width = I1.cols;
       const int height = I1.rows;
       const int src_depth = I1.type() == CV_8U ? 8 : 16;
       const int src_bytes = src_depth * width * height / 8;
       int dst_depth;
       // Penalization parameter P1
       const int p1 = parser.get<int>("p1");
       // Smoothing Parameter P2
       const int p2 = parser.get<int>("p2");
       // Uniqueness Parameter 
       const float unq = 0.95f;
       // Sup Pixel precision 
       const bool subp = parser.get<bool>("sub_p");
       const int path_type = parser.get<int>("path_type");
       const int cens_t = parser.get<int>("census_type");
       if(subp)
       {
                dst_depth = 16;
       }else
       {
                dst_depth = disp_size < 256 ? 8 : 16;
       }
       const int dst_bytes = dst_depth * width * height / 8;
       // min Disparity value
       const int min_d_v = 0;
       // max differences from left and right 
       const int max_d_v = 1;
       // symmetric census 1 for sgm::CensusType::SYMMETRIC_CENSUS_9x7 else 0 sgm::CensusType::CENSUS_9x7
       const sgm::CensusType census_t = cens_t == 1 ? sgm::CensusType::SYMMETRIC_CENSUS_9x7 : sgm::CensusType::CENSUS_9x7;
       // number of Path : 4 , if 8 do sgm::PathType::SCAN_8PATH
       const sgm::PathType path = path_type == 0 ? sgm::PathType::SCAN_4PATH : sgm::PathType::SCAN_8PATH;
       // Initialize the Celsus SGM Paraemters
       const sgm::StereoSGM::Parameters param(p1, p2, unq, subp, path, min_d_v, max_d_v, census_t);
       // Initialize the SGM Class
       sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
       // Define the GPU Buffer for Left,  Right and Final Disparity 
       device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
       cv::Mat disparity(height, width, dst_depth == 8 ? CV_8S : CV_16S), disparity_color;
       // Obtain the Invalid dispairty value
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
       std::cout << "SGM Execution Time : " << 1e-3 * duration << " [msec] " << fps << "  [FPS]" << std::endl;
       // Store the Results as PFM file
       std::cout << "2 - Write the output as PFM file " << std::endl;
       // Initialize the scaling of the disparity, 1 if not subpixel precision 16 otherwise
       const int d_scale = subp ? sgm::StereoSGM::SUBPIXEL_SCALE : 1; 
       const int disp_s = d_scale * disp_size;
       // Retrieve the mask of Invalid disparity
       cv::Mat disp_mask(disparity.rows, disparity.cols, disparity.type(), disparity.data);
       cv::Mat inv_mask = disp_mask == sgm.get_invalid_disparity();
       // Set to 0 the invalid values
       if(!inv_mask.empty() && disparity.size() == inv_mask.size())
       {
                disparity.setTo(0, inv_mask); 
       }
       // Convert the Disparity to 32FC1 and scaling
       disparity.convertTo(disparity, CV_32FC1, 1.0);
       /* In order to retrive the correct disparity due the downscaling we 
           1 - Resize with Nearest Neighbour Interlinear,
           2 - Retrieve the Scale Factor = original size width / (float) Kitti size width
           3 - Then scale the disparity as Disparity Map * Scale Factor
       */
       if(reduce)
       {
                cv::resize(disparity,disparity, output_size, cv::INTER_NEAREST);
                float res_scale_w = output_size.width/(float)down_width;
                float res_scale = res_scale_w;
                disparity *= res_scale;
       }
       // Store the Disparity as a .pfm file
       savePFM(disparity, disp_path);
     return 0;
}
