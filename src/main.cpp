#include <iostream>
#include <filesystem>

#include "../include/tensor.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

string WAREHOUSE_PATH = "data/FaceWarehouse/";
string CORE_TENSOR_PATH = "data/core_tensor.bin";

int main() {
    // Core tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 coreTensor(150, 47, 11510);

    // If core tensor file does not exist, build it from face warehouse
//    if (std::filesystem::exists(CORE_TENSOR_PATH)) {
//        loadCoreTensor(CORE_TENSOR_PATH, coreTensor);
//    }
//    else {
//        buildCoreTensor(WAREHOUSE_PATH, CORE_TENSOR_PATH, coreTensor);
//    }


    string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
    string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
    vector<cv::Point2f> lms = readLandmarksFromFile_2(land_path, image);

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
