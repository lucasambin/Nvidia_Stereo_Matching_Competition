#include "sample_common.h"

#include <opencv2/imgproc.hpp>

void colorize_disparity(const cv::Mat& src, cv::Mat& dst, int disp_size, cv::InputArray mask)
{
	cv::Mat tmp;
	src.convertTo(tmp, CV_8U, 255. / disp_size);
	cv::applyColorMap(tmp, dst, cv::COLORMAP_JET);

	if (!mask.empty())
		dst.setTo(0, mask);
}
