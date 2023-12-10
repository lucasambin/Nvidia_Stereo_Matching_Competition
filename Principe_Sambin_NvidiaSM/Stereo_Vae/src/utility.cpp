#include "utility.h"

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

// checkPath takes as input the current path and completes it , if is not a valid path returns -1
void checkPath(int &flag,const std::string path, std::string &final_path)
{
    flag = 0;
    // Check the path given in input , if contains as last string \ or / then it will be the final path,
    // else if the path does not contains at the final character \ or / search the first occurency in the string 
    // and adds it to form the final path or if does not contains any \ or / not save anything
    if(path[path.size()-1] == '/')
    {
        final_path = path;  
    }else if(path[path.size()-1] == '\\'){
        final_path = path; 
    }else{
        size_t found = path.find_first_of("\\");
        if(found != cv::String::npos)
        {
            final_path = path+"\\";       
        }else{
            size_t found = path.find_first_of('/');
            if(found != cv::String::npos)
            {
                final_path = path+"/"; 
            }else{
                std::cout << "Not a full path , rewrite the full path where the files will be then stored" << std::endl;
                flag = -1;
            }
        }
    }
}

// retrieveImage obtain the images from a folder, reads them and saves them into a vector of images
int retrieveImage(size_t count,std::vector<cv::String> file_name,std::vector<cv::Mat> &images)
{
    // Check if there are images or exit
	if(count == 0)
	{
		std::cout << "No Image Files in the folder chosen." << std::endl;
		return -1;
	}
    // Retriving the images
	for(int i=0;i<count;i++){
		cv::Mat img=cv::imread(file_name[i], cv::IMREAD_GRAYSCALE);
		if(!img.empty()){
		    images.push_back(img);
		}
		else{
			return -1;
		}
	}
    return 0;
}

// saveImg saves the images given in input from a vector to a specified folder in .jpg format
void saveImg(const std::vector<cv::Mat> img, const cv::String path)
{
    for(int i=0 ; i<img.size();i++)
    {
        // derive the current image name
        cv::String current_img = "0";
        // if the image is less than the 10th write the current indx with a 0 before
        // otherwise write just the current indx 
        if(i<9)
        {
            current_img = '0'+std::to_string(i+1);
        }else{
            current_img = std::to_string(i+1);
        }
        // Check the path given in input , if contains as last string \ or / adds just the name of the file.jpg and save it
        // else if the path does not contains at the final character \ or / search the first occurency in the string 
        // and adds it then save the current image or if does not contains any \ or / not save anything
        if(path[path.size()-1] == '/')
        {
            cv::imwrite(path+"/"+current_img+".png",img[i]);
        }else if(path[path.size()-1] == '\\'){
            cv::imwrite(path+"\\"+current_img+".png",img[i]);
        }else{
            size_t found = path.find_first_of("\\");
            if(found != cv::String::npos)
            {
                cv::imwrite(path+"\\"+current_img+".png",img[i]);
            }else{
                size_t found = path.find_first_of('/');
                if(found != cv::String::npos)
                {
                    cv::imwrite(path+"/"+current_img+".png",img[i]);
                }else{
                    std::cout << "Not a full path , rewrite the full path where the files will be then stored" << std::endl;
                }
            }
        }
    }
}