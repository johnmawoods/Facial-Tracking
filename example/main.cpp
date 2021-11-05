#include "utility.h"

#include <filesystem>
#include "opencv2/calib3d/calib3d.hpp"

namespace fs = std::filesystem;

vector<cv::Point3d> loadGenModel(const std::string& path) {
	vector<cv::Point3d> gen3dModel(68);
	std::ifstream infile(path);  
	if (infile.fail()) {
		std::cerr << "ERROR: couldn't open the shape 3d file to read the 3D face points from" << endl;
		exit(-1);
	}
	for (int i = 0; i < 68; i++) {
		std::string temp;
		std::getline(infile, temp, ' ');
		gen3dModel[i].x = std::stof(temp);
		std::getline(infile, temp, ' ');
		gen3dModel[i].y = std::stof(temp);
		std::getline(infile, temp);
		gen3dModel[i].z = std::stof(temp);
	}
	infile.close();
	return gen3dModel;
}


int main() {

    //--------------------------- load inputs
    //======================================================================

	vector<cv::Point3d> gen3dModel = loadGenModel("../data/shape3d.pts");   // an arbitrary generic sparse 3D face model with 68 vertices
	cv::Mat image = cv::imread("../data/pose_1.png", 1);
	asu::Utility util;
	vector<cv::Point2f> lms = util.readLandmarksFromFile("../data/pose_1.land", image);

    //----- select a few representative points for pose estimation
    //======================================================================
    
    /*left eye left corner, left eye right corner, right eye left corner, right eye right corner, 
	 nose tip, mouth left corner, mouth right corner*/
	vector<int> indices_2d = { 27, 31, 35, 39, 54, 55, 61 };   // 2d landmark indices
	vector<int> indices_3d = { 36, 39, 42, 45, 30, 48, 54 };   // corresponding 3d vertex indices from the generic model
    
    int numEstimationPoints = indices_3d.size();
    vector<cv::Point2f> lmsVec(numEstimationPoints);
    vector<cv::Point3f> objectVec(numEstimationPoints);
    for (int i = 0; i < numEstimationPoints; i++) {
        lmsVec[i] = lms[indices_2d[i]];
        objectVec[i] = gen3dModel[indices_3d[i]];
	}

    //--------------------- Aatmik's code (slightly modified)
    //======================================================================

    //double fx = 640, fy = 640, cx = 320, cy = 240;

    double f = image.cols;               // ideal camera where fx ~ fy
    double cx = image.cols / 2.0;
    double cy = image.rows / 2.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
     
    //// Assuming no distortion           // ideal camera no distortion
    //cv::Mat distCoeffs(4, 1, CV_64F);
    //distCoeffs.at<double>(0) = 0;
    //distCoeffs.at<double>(1) = 0;
    //distCoeffs.at<double>(2) = 0;
    //distCoeffs.at<double>(3) = 0;   

    // Get rotation and translation parameters
    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);
    //cv::solvePnP(objectVec, lmsVec, cameraMatrix, distCoeffs, rvec, tvec);
    cv::solvePnP(objectVec, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);

    // Convert Euler angles to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
    T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
    // Transform object
    std::vector<cv::Mat> cameraVec;
    for (auto& vec : objectVec) {
        double data[4] = { vec.x, vec.y, vec.z, 1 };
        cv::Mat vector4d = cv::Mat(4, 1, CV_64F, data);
        cv::Mat result = T * vector4d;
        cameraVec.push_back(result);
    }

    // Project points onto image
    std::vector<cv::Point2f> imageVec;
    for (auto& vec : cameraVec) {
        cv::Point2f result;
        //result.x = fx * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        //result.y = fx * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;

		result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
		result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;

        imageVec.push_back(result);
    }
    //    cv::projectPoints(objectVec, rvec, tvec, cameraMatrix, distCoeffs, imageVec);


    cv::Mat visualImage = image.clone();       // deep copy of the image to avoid manipulating the image itself
    //cv::Mat visualImage = image;             // shallow copy
    double sc = 2;
    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
    for (int i = 0; i < imageVec.size(); i++) {
        //cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
        //cv::putText(visualImage, std::to_string(i), imageVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

        cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
        cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)

        //cv::putText(visualImage, std::to_string(i), imageVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

    }
    cv::imshow("visualImage", visualImage);
    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);

	return 0;
}