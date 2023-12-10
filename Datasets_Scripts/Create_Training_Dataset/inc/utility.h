#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/contrib/contrib.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/version.hpp>

bool savePFM(const cv::Mat image, const std::string filePath);
void checkPath(int &flag,const std::string path, std::string &final_path);
int retrieveImage(size_t count,std::vector<cv::String> file_name,std::vector<cv::Mat> &images);
void saveImg(const std::vector<cv::Mat> img, const cv::String path);

#endif // !__UTILITY_H__
