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
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

string WAREHOUSE_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/FaceWarehouse/";
string RAW_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/shape_tensor.bin";


int main() {
    // Raw tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 rawTensor(150, 47, 11510);
    cout << "rawTensor template created" << endl;
    
    tensor3 shapeTensor(150, 47, 73);
<<<<<<< HEAD
    cout << "shapeTensor template created" << endl;
    
    cout << "checkForTensors" << endl;
    // If raw tensor file does not exist, build it from face warehouse
=======


>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba
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
<<<<<<< HEAD
    
    // Load or build shape tensor
=======

>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba
    if (std::filesystem::exists(SHAPE_TENSOR_PATH)) {
        cout << "shape tensor exists" << endl;
        loadShapeTensor(SHAPE_TENSOR_PATH, shapeTensor);
<<<<<<< HEAD
        cout << "shape tensor loaded" << endl;
    }
    else {
        cout << "shape tensor does not exist" << endl;
=======
    } else {
>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba
        buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
        cout << "shape tensor built" << endl;
    }
<<<<<<< HEAD
    cout << "create test.obj" << endl;
    std::ofstream file ("D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/test.obj");
    for (int k = 0; k < 73; k++) {
        Eigen::Vector3f v = shapeTensor(0, 0, k);
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    file.close();
    
=======

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector
    int n_vectors = 73;
    std::vector<cv::Point3f> objectVec(n_vectors);
    for (int i = 0; i < n_vectors; ++i) {
        Eigen::Vector3f eigen_vec = shapeTensor(0, 0, i);
        cv::Point3f cv_vec;
        cv_vec.x = eigen_vec.x();
        cv_vec.y = eigen_vec.y();
        cv_vec.z = eigen_vec.z();
        objectVec[i] = cv_vec;
    }


    // Image vector contains 2d landmark positions
    string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
    string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
//    asu::Utility util;
    vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image);

    double f = image.cols;               // ideal camera where fx ~ fy
    double cx = image.cols / 2.0;
    double cy = image.rows / 2.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    // Get rotation and translation parameters
    cv::Mat rvec(3, 1, CV_64F);
    cv::Mat tvec(3, 1, CV_64F);
    cv::solvePnP(objectVec, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);

    // Convert Euler angles to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
    T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
    cout << tvec << endl << rvec << endl;

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
        result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
        result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;

        imageVec.push_back(result);
    }
    cout << T << endl << cameraVec[0] << endl;
    cv::projectPoints(objectVec, rvec, tvec, cameraMatrix, cv::Mat(), imageVec);
>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba

    /*
    std::string img_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.png";
    std::string land_path = WAREHOUSE_PATH + "Tester_103/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
    vector<cv::Point2f> lms = readLandmarksFromFile(land_path, image);

<<<<<<< HEAD
    cv::Mat visualImage = image.clone();       // deep copy of the image to avoid manipulating the image itself
    //cv::Mat visualImage = image;             // shallow copy
    float sc = 1;
    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
    for (int i = 0; i < lms.size(); i++) {
        cv::circle(visualImage, lms[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
        cv::putText(visualImage, std::to_string(i), lms[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);
=======
    cv::Mat visualImage = image.clone();
    double sc = 3;
    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
    for (int i = 0; i < imageVec.size(); i++) {
        //cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
//        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

        cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
//        cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)

//        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);

>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba
    }
    cv::imshow("visualImage", visualImage);
    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);
<<<<<<< HEAD
    */
}
=======

    return 0;


}


//    std::ofstream file ("test.obj");
//    for (int k = 0; k < 73; k++) {
//        Eigen::Vector3f v = shapeTensor(0, 0, k);
//        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
//    }
//    file.close();

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


>>>>>>> 7c0c8c872bf4729235c42e070364040b23bee6ba
