/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for colormap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring> // for memset
#include <iostream>
#include <fstream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

bool savePFM(const cv::Mat image, const std::string filePath)
{
    //Open the file as binary!
    std::ofstream imageFile(filePath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    if(imageFile)
    {
        int width(image.cols), height(image.rows);
        int numberOfComponents(image.channels());

        //Write the type of the PFM file and ends by a line return
        char type[3];
        type[0] = 'P';
        type[2] = 0x0a;

        if(numberOfComponents == 3)
        {
            type[1] = 'F';
        }
        else if(numberOfComponents == 1)
        {
            type[1] = 'f';
        }

        imageFile << type[0] << type[1] << type[2];

        //Write the width and height and ends by a line return
        imageFile << width << " " << height << type[2];

        //Assumes little endian storage and ends with a line return 0x0a
        //Stores the type
        char byteOrder[10];
        byteOrder[0] = '-'; byteOrder[1] = '1'; byteOrder[2] = '.'; byteOrder[3] = '0';
        byteOrder[4] = '0'; byteOrder[5] = '0'; byteOrder[6] = '0'; byteOrder[7] = '0';
        byteOrder[8] = '0'; byteOrder[9] = 0x0a;

        for(int i = 0 ; i<10 ; ++i)
        {
            imageFile << byteOrder[i];
        }

        //Store the floating points RGB color upside down, left to right
        float* buffer = new float[numberOfComponents];

        for(int i = 0 ; i<height ; ++i)
        {
            for(int j = 0 ; j<width ; ++j)
            {
                if(numberOfComponents == 1)
                {
                    buffer[0] = image.at<float>(height-1-i,j);
                }
                else
                {
                    cv::Vec3f color = image.at<cv::Vec3f>(height-1-i,j);

                   //OpenCV stores as BGR
                    buffer[0] = color.val[2];
                    buffer[1] = color.val[1];
                    buffer[2] = color.val[0];
                }

                //Write the values
                imageFile.write((char *) buffer, numberOfComponents*sizeof(float));

            }
        }

        delete[] buffer;

        imageFile.close();
    }
    else
    {
        std::cerr << "Could not open the file : " << filePath << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImageLeft, cvImageRight;

    // VPI objects that will be used
    VPIImage inLeft        = NULL;
    VPIImage inRight       = NULL;
    VPIImage tmpLeft       = NULL;
    VPIImage tmpRight      = NULL;
    VPIImage stereoLeft    = NULL;
    VPIImage stereoRight   = NULL;
    VPIImage disparity     = NULL;
    VPIImage confidenceMap = NULL;
    VPIStream stream       = NULL;
    VPIPayload stereo      = NULL;

    int retval = 0;
    
    // Timing Variables 
    VPIEvent evStart = NULL;
    VPIEvent evPrePro = NULL;
    VPIEvent evSM = NULL;
    VPIEvent evTotal = NULL;
    
    try
    {
        // =============================
        // Parse command line parameters

        if (argc != 5)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cpu|pva|cuda|pva-nvenc-vic> <left image> <right image> <ouput pfm file>");
        }

        std::string strBackend       = argv[1];
        std::string strLeftFileName  = argv[2];
        std::string strRightFileName = argv[3];
        std::string outputFileName = argv[4];
        uint32_t backends;

        if (strBackend == "cpu")
        {
            backends = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backends = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backends = VPI_BACKEND_PVA;
        }
        else if (strBackend == "pva-nvenc-vic")
        {
            backends = VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu, cuda, pva or pva-nvenc-vic.");
        }

        // =====================
        // Load the input images
        cvImageLeft = cv::imread(strLeftFileName);
        if (cvImageLeft.empty())
        {
            throw std::runtime_error("Can't open '" + strLeftFileName + "'");
        }

        cvImageRight = cv::imread(strRightFileName);
        if (cvImageRight.empty())
        {
            throw std::runtime_error("Can't open '" + strRightFileName + "'");
        }

        // =================================
        // Allocate all VPI resources needed

        int32_t inputWidth  = cvImageLeft.cols;
        int32_t inputHeight = cvImageLeft.rows;

        // Create the stream that will be used for processing.
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageLeft, 0, &inLeft));
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageRight, 0, &inRight));

        // Format conversion parameters needed for input pre-processing
        VPIConvertImageFormatParams convParams;
        CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

        // Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
        VPIStereoDisparityEstimatorCreationParams stereoParams;
        CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams));

        // Define some backend-dependent parameters

        VPIImageFormat stereoFormat;
        int stereoWidth, stereoHeight;
        if (strBackend == "pva-nvenc-vic")
        {
            stereoFormat = VPI_IMAGE_FORMAT_Y16_ER_BL;

            // Input width and height has to be 1920x1080 in block-linear format for pva-nvenc-vic pipeline
            stereoWidth  = 1920;
            stereoHeight = 1080;

            // For PVA+NVENC+VIC mode, 16bpp input must be MSB-aligned, which
            // is equivalent to say that it is Q8.8 (fixed-point, 8 decimals).
            convParams.scale = 256;

            // Maximum disparity is fixed to 256.
            stereoParams.maxDisparity = 64;
        }
        else
        {
            stereoFormat = VPI_IMAGE_FORMAT_Y16_ER;

            if (strBackend == "pva")
            {
                stereoWidth  = 480;
                stereoHeight = 270;
            }
            else
            {
                stereoWidth  = inputWidth;
                stereoHeight = inputHeight;
            }

            stereoParams.maxDisparity = 54;
        }

        // Create the payload for Stereo Disparity algorithm.
        // Payload is created before the image objects so that non-supported backends can be trapped with an error.
        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backends, stereoWidth, stereoHeight, stereoFormat, &stereoParams,
                                                       &stereo));

        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, VPI_IMAGE_FORMAT_U16, 0, &disparity));

        if (strBackend == "pva-nvenc-vic")
        {
            // Need an temporary image to convert BGR8 input from OpenCV into pixel-linear 16bpp grayscale.
            // We can't convert it directly to block-linear since CUDA backend doesn't support it, and
            // VIC backend doesn't support BGR8 inputs.
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpLeft));
            CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmpRight));

            // Input to pva-nvenc-vic stereo disparity must be block linear
            CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoLeft));
            CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoRight));

            // confidence map is needed for pva-nvenc-vic pipeline
            CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
        }
        else
        {
            // PVA requires that input resolution is 480x270
            if (strBackend == "pva")
            {
                CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpLeft));
                CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, stereoFormat, 0, &tmpRight));
            }
            else if (strBackend == "cuda")
            {
                CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));
            }

            // Allocate input to stereo disparity algorithm, pitch-linear 16bpp grayscale
            CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoLeft));
            CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoRight));
        }

        // ================
        // Processing stage
        // -----------------
        // Pre-process input
        // Create the events we'll need to get timing info
        CHECK_STATUS(vpiEventCreate(0, &evStart));
        CHECK_STATUS(vpiEventCreate(0, &evPrePro));
        CHECK_STATUS(vpiEventCreate(0, &evSM));
        //CHECK_STATUS(vpiEventCreate(0, &evTotal));
        // Record stream queue when we start processing
        CHECK_STATUS(vpiEventRecord(evStart, stream));
        if (strBackend == "pva-nvenc-vic" || strBackend == "pva")
        {
            // Convert opencv input to temporary grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, tmpLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, tmpRight, &convParams));

            // Do both scale and final image format conversion on VIC.
            CHECK_STATUS(
                vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpLeft, stereoLeft, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
            CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_VIC, tmpRight, stereoRight, VPI_INTERP_LINEAR,
                                          VPI_BORDER_CLAMP, 0));
        }
        else
        {
            // Convert opencv input to grayscale format using CUDA
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, stereoLeft, &convParams));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, stereoRight, &convParams));
        }
        // Record stream queue when we ending PreProcessing
        CHECK_STATUS(vpiEventRecord(evPrePro, stream));
        // ------------------------------
        // Do stereo disparity estimation

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, backends, stereo, stereoLeft, stereoRight, disparity,
                                                       confidenceMap, NULL));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));
        // Record stream queue when we ending PreProcessing
        CHECK_STATUS(vpiEventRecord(evSM, stream));
        // ========================================
        // Output pre-processing and saving to disk
        // Lock output to retrieve its data on cpu memory
        VPIImageData data;
        CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

        // Make an OpenCV matrix out of this image
        cv::Mat cvDisparity;
        CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));
    
        // Scale result and write it to disk. Disparities are in Q10.5 format (CV_16UC1 type 2),
        // so to map it to float, it gets divided by 32. Then the resulting disparity range,
        // from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output
        
        double max_v;
        double min_v;
        cv::Point min_loc;
        cv::Point max_loc;
        //cvDisparity.convertTo(cvDisparity, CV_32FC1, 1/32, 0);
        //cvDisparity.convertTo(cvDisparity, CV_32FC1, 1 / (32 * stereoParams.maxDisparity), 0);
        std::cout << "Disp Map Type : " << cvDisparity.type() << std::endl;
        cv::minMaxLoc(cvDisparity, &min_v, &max_v, &min_loc, &max_loc);
        double num_disp = max_v - min_v;
        std::cout << "B Max Disp : " << max_v << std::endl;
        std::cout << "B Min Disp : " << min_v << std::endl;
        std::cout << "B Num Disp : " << num_disp << std::endl;
        //std::cout << "Disp Map : " << cvDisparity << std::endl;

        cvDisparity.convertTo(cvDisparity, CV_32FC1, 1, 0);
        cvDisparity = cvDisparity/32;
        cv::minMaxLoc(cvDisparity, &min_v, &max_v, &min_loc, &max_loc);
        num_disp = max_v - min_v;
        std::cout << "A Max Disp : " << max_v << std::endl;
        std::cout << "A Min Disp : " << min_v << std::endl;
        std::cout << "A Num Disp : " << num_disp << std::endl;
        
        std::cout << "Final Disp Map Type : " << cvDisparity.type() << std::endl;
        //std::cout << "Disp Map : " << cvDisparity << std::endl;
        //cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 /(stereoParams.maxDisparity), 0);
        
        //cv::imwrite("disparity_" + strBackend + ".png", cvDisparity);
        savePFM(cvDisparity, outputFileName);
        CHECK_STATUS(vpiImageUnlock(disparity));
        // Apply JET colormap to turn the disparities into color, reddish hues
        // represent objects closer to the camera, blueish are farther away.
        //CHECK_STATUS(vpiEventCreate(0, &evTotal));
        //CHECK_STATUS(vpiEventRecord(evTotal, stream));
        
        // 3. Timing analysis ----------------------
  
        float elapsedPreMS, elapsedSMMS, elapsedTotalMS, fps;
        CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evPrePro, &elapsedPreMS));
        CHECK_STATUS(vpiEventElapsedTimeMillis(evPrePro, evSM, &elapsedSMMS));
        CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evSM, &elapsedTotalMS));
        fps = 1e3 / elapsedTotalMS;
        printf("Total elapsed time: %f [ms]\t  %f [fps]\n", elapsedTotalMS, fps);
        
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // ========
    // Clean up

    // Destroying stream first makes sure that all work submitted to
    // it is finished.
    vpiStreamDestroy(stream);

    // Only then we can destroy the other objects, as we're sure they
    // aren't being used anymore.

    vpiImageDestroy(inLeft);
    vpiImageDestroy(inRight);
    vpiImageDestroy(tmpLeft);
    vpiImageDestroy(tmpRight);
    vpiImageDestroy(stereoLeft);
    vpiImageDestroy(stereoRight);
    vpiImageDestroy(confidenceMap);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);

    return retval;
}

// vim: ts=8:sw=4:sts=4:et:ai
