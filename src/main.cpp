#include <iostream>
#include <filesystem>

#include "../include/tensor.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

string WAREHOUSE_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/FaceWarehouse/";
string RAW_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/shape_tensor.bin";


int main() {
    // Raw tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 rawTensor(150, 47, 11510);
    cout << "rawTensor template created" << endl;
    
    tensor3 shapeTensor(150, 47, 73);
    cout << "shapeTensor template created" << endl;
    
    cout << "checkForTensors" << endl;
    // If raw tensor file does not exist, build it from face warehouse
    if (std::filesystem::exists(RAW_TENSOR_PATH)) {
        cout << "raw tensor exists" << endl;
        loadRawTensor(RAW_TENSOR_PATH, rawTensor);
        cout << "raw tensor loaded" << endl;
    }
    else {
        cout << "raw tensor does not exist" << endl;
        buildRawTensor(WAREHOUSE_PATH, RAW_TENSOR_PATH, rawTensor);
        cout << "raw tensor built" << endl;
    }
    
    // Load or build shape tensor
    if (std::filesystem::exists(SHAPE_TENSOR_PATH)) {
        cout << "shape tensor exists" << endl;
        loadShapeTensor(SHAPE_TENSOR_PATH, shapeTensor);
        cout << "shape tensor loaded" << endl;
    }
    else {
        cout << "shape tensor does not exist" << endl;
        buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
        cout << "shape tensor built" << endl;
    }
    cout << "create test.obj" << endl;
    std::ofstream file ("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/test.obj");
    for (int k = 0; k < 73; k++) {
        Eigen::Vector3f v = shapeTensor(0, 0, k);
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    file.close();
    

    /*
    std::string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
    std::string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
    vector<cv::Point2f> lms = readLandmarksFromFile(land_path, image);

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
    */
}
