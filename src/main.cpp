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
string RAW_TENSOR_PATH = "data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "data/shape_tensor.bin";

int main() {
    // Raw tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 rawTensor(150, 47, 11510);
    tensor3 shapeTensor(150, 47, 73);

    // If raw tensor file does not exist, build it from face warehouse
    if (std::filesystem::exists(RAW_TENSOR_PATH)) {
        loadRawTensor(RAW_TENSOR_PATH, rawTensor);
    }
    else {
        buildRawTensor(WAREHOUSE_PATH, RAW_TENSOR_PATH, rawTensor);
    }

    // Load or build shape tensor
    if (std::filesystem::exists(SHAPE_TENSOR_PATH)) {
        loadShapeTensor(SHAPE_TENSOR_PATH, shapeTensor);
    }
    else {
        buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
    }

    std::ofstream file ("test.obj");
    for (int k = 0; k < 73; k++) {
        Eigen::Vector3f v = shapeTensor(0, 0, k);
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    file.close();


//    string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
//    string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
//    cv::Mat image = cv::imread(img_path, 1);
//    vector<cv::Point2f> lms = readLandmarksFromFile_2(land_path, image);
//
//    cv::Mat visualImage = image.clone();       // deep copy of the image to avoid manipulating the image itself
//    //cv::Mat visualImage = image;             // shallow copy
//    float sc = 1;
//    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
//    for (int i = 0; i < lms.size(); i++) {
//        cv::circle(visualImage, lms[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
//        cv::putText(visualImage, std::to_string(i), lms[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);
//    }
//    cv::imshow("visualImage", visualImage);
//    int key = cv::waitKey(0) % 256;
//    if (key == 27)                        // Esc button is pressed
//        exit(1);

}
