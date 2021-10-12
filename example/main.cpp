#include "utility.h"

#include <filesystem>

namespace fs = std::filesystem;

int main() {

	asu::Utility util;

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
			vector<cv::Point2f> lms = util.readLandmarksFromFile(lmsFileName, image);   
			
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