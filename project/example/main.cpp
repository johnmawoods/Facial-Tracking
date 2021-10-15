#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

namespace fs = std::filesystem;

using std::cout;
using std::endl;
using std::vector;


vector<cv::Point2f> readLandmarksFromFile_1(const std::string& path, const cv::Mat& image) {

	std::ifstream infile(path);
	if (infile.fail()) {
		std::cerr << "ERROR: unable to open the landmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
		exit(-1);
	}
	std::string hay;
	std::getline(infile, hay);
	int nPoints = std::stof(hay);
	vector<cv::Point2f> lms(nPoints * 2);
	for (int i = 0; i < nPoints; i++) {
		std::string temp;
		std::getline(infile, temp, ' ');
		lms[i].x = std::stof(temp) * image.cols;
		std::getline(infile, temp);
		lms[i].y = image.rows - (std::stof(temp) * image.rows);
	}
	infile.close();
	
	return lms;

}


vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image) {

	vector<int> orderedIndices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  //face contour
					   21, 22, 23, 24, 25, 26,                            //left eyebrow
					   18, 17, 16, 15, 20, 19,                            //right eyebrow
					   27, 66, 28, 69, 29, 68, 30, 67,                    //left eye 
					   33, 70, 32, 73, 31, 72, 34, 71,                    //right eye 
					   35, 36, 37, 38, 44, 39, 45, 40, 41, 42, 43,        //nose contour 
					   65,												  //nose tip
					   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,    //outer mouth
					   63, 62, 61, 60, 59, 58						      //inner mouth
	};

	std::ifstream infile(path);
	if (infile.fail()) {
		std::cerr << "ERROR: unable to open the landmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
		exit(-1);
	}
	std::string hay;
	std::getline(infile, hay);
	int nOrigPoints = std::stof(hay);
	vector<float> origLms(nOrigPoints * 2);
	for (int i = 0; i < nOrigPoints; i++) {
		std::string temp;
		std::getline(infile, temp, ' ');
		origLms[i] = std::stof(temp) * image.cols;
		std::getline(infile, temp);
		origLms[i + nOrigPoints] = image.rows - (std::stof(temp) * image.rows);
	}
	infile.close();

	int nPoints = orderedIndices.size();
	vector<cv::Point2f> lms(nPoints * 2);
	for (int i = 0; i < nPoints; i++) {
		lms[i].x = origLms[orderedIndices[i]];
		lms[i].y = origLms[orderedIndices[i] + nOrigPoints];
	}

	return lms;

}



int main() {

	//-------- capturing video
	//==================================

	int cameraID = 0;
	cv::VideoCapture cap(cameraID);                   // get frames from webcame

//	std::string videoname = "demo.mp4";               // get frames from video
//	cv::VideoCapture cap(videoname);
	if (!cap.isOpened()) {
		std::cerr << "ERROR: unable to connect to the video" << endl;
		exit(-1);
	}

	cv::Mat frame;

	while(cap.read(frame)){
		cv::imshow("frame", frame);
		int key = cv::waitKey(1) % 256;
		if(key == 27)                        // Esc button is pressed
			break;
	}



	std::string path = "../data/";

	for (fs::directory_iterator topIter(path); topIter != fs::directory_iterator(); topIter++) {

		std::string foldername = topIter->path().stem().string();
		cout << endl << "============ " << foldername << " ============" << endl << endl;
		std::string nextPath = topIter->path().string() + "/TrainingPose/";
		int idx = 0;

		for (fs::directory_iterator iter(nextPath); iter != fs::directory_iterator(); iter++) {

			std::string extension = iter->path().extension().string();
			if (!(extension == ".png" || extension == ".jpg"))
				continue;

			std::string stemName = iter->path().stem().string();
			cout << "idx = " << idx++ << " --> " << stemName << endl;

			cv::Mat image = cv::imread(iter->path().string(), 1);

			std::string lmsFileName = nextPath + stemName + ".land";
			vector<cv::Point2f> lms = readLandmarksFromFile_1(lmsFileName, image);   
			//vector<cv::Point2f> lms = readLandmarksFromFile_2(lmsFileName, image);   
			
			//---------------- visualize to check if everything is in place correctly 
			//==========================================================================================

			cv::Mat visualImage = image.clone();       // deep copy of the image to avoid manipulating the image itself
			//cv::Mat visualImage = image;             // shallow copy
			float sc = 1;
			cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
			for (int i = 0; i < lms.size(); i++) {
				cv::circle(visualImage, lms[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
				cv::putText(visualImage, std::to_string(i), lms[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);
			}
			cv::imshow("visualImage", visualImage);
			int key = cv::waitKey(0) % 256;
			if (key == 27)                        // Esc button is pressed      
				exit(1);

		}

	}



	return 0;
}