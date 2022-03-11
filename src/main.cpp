#include <iostream>
#include <filesystem>
#include <vector>
#include <random>

#include "ceres/ceres.h"

#include "../include/tensor.h"
#include "../include/optimization.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>

string WAREHOUSE_PATH = "C:/Users/stefa/Desktop/Capstone/repo/Facial-Tracking/data/FaceWarehouse/";
string RAW_TENSOR_PATH = "C:/Users/stefa/Desktop/Capstone/repo/Facial-Tracking/data/raw_tensor.bin";
string SHAPE_TENSOR_PATH = "C:/Users/stefa/Desktop/Capstone/repo/Facial-Tracking/data/shape_tensor.bin";


int main() {
    // Raw tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 rawTensor(150, 47, 11510);
    tensor3 shapeTensor(150, 47, 73);

    if (std::filesystem::exists(RAW_TENSOR_PATH)) {
        loadRawTensor(RAW_TENSOR_PATH, rawTensor);
    }
    else {
        buildRawTensor(WAREHOUSE_PATH, RAW_TENSOR_PATH, rawTensor);
    }

    if (std::filesystem::exists(SHAPE_TENSOR_PATH)) {
        loadShapeTensor(SHAPE_TENSOR_PATH, shapeTensor);
    }
    else {
        buildShapeTensor(rawTensor, SHAPE_TENSOR_PATH, shapeTensor);
    }


    // Image vector contains 2d landmark positions
    string img_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.png";
    string land_path = WAREHOUSE_PATH + "Tester_138/TrainingPose/pose_1.land";
    cv::Mat image = cv::imread(img_path, 1);
    // "ground truth 2d landmarks"
    vector<cv::Point2f> lmsVec = readLandmarksFromFile_2(land_path, image);

    double f = image.cols;               // ideal camera where fx ~ fy
    double cx = image.cols / 2.0;
    double cy = image.rows / 2.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);

    // Get rotation and translation parameters
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);


    // vector representing 47 expressions
    // set to neutral expression

    int numExpressions = 47;
    Eigen::VectorXf w(numExpressions);

    for (int i = 0; i < numExpressions; i++)
    {
        w[i] = 0;
    }
    w[0] = 1;

    /** Transform from object coordinates to camera coordinates **/
    // Copy Eigen vector to OpenCV vector
    int n_vectors = 73;
    std::vector<cv::Point3f> singleExp(n_vectors);
    std::vector<std::vector<cv::Point3f>> multExp(numExpressions);
    // 47 expressions
    for (int j = 0; j < numExpressions; j++)
    {
        // 73 vertices
        for (int i = 0; i < 73; i++)
        {
            Eigen::Vector3f tens_vec = shapeTensor(137, j, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            singleExp[i] = conv_vec;
        }
        multExp[j] = singleExp;
    }

    std::vector<cv::Point3f> combinedExp(n_vectors);


    
    // create new face based on weights
    for (int i = 0; i < 73; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {

            combinedExp[i].x = combinedExp[i].x + (multExp[j][i].x * w[j]);
            combinedExp[i].y = combinedExp[i].y + (multExp[j][i].y * w[j]);
            combinedExp[i].z = combinedExp[i].z + (multExp[j][i].z * w[j]);
        }
    }


    //pose estimation
    cv::solvePnP(combinedExp, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);

    std::vector<float> poseVec(6, 0);

    poseVec[0] = rvec.at<double>(0);
    poseVec[1] = rvec.at<double>(1);
    poseVec[2] = rvec.at<double>(2);

    poseVec[3] = tvec.at<double>(0);
    poseVec[4] = tvec.at<double>(1);
    poseVec[5] = tvec.at<double>(2);

    // optimization
    optimize(lmsVec, poseVec, image, f, w);

    
    
    int numDenseVerts = 11510;
    std::vector<cv::Point3f> denseSingleExp(numDenseVerts);
    std::vector<std::vector<cv::Point3f>> denseMultExp(numExpressions);
    // 47 expressions
    for (int j = 0; j < numExpressions; j++)
    {
        cout << j << endl;
        for (int i = 0; i < numDenseVerts; i++)
        {
           
            Eigen::Vector3f tens_vec = rawTensor(137, j, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            denseSingleExp[i] = conv_vec;
        }
        denseMultExp[j] = denseSingleExp;
    }
    
    std::vector<cv::Point3f> denseCombinedExp(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            denseCombinedExp[i].x = denseCombinedExp[i].x + (denseMultExp[j][i].x * w[j]);
            denseCombinedExp[i].y = denseCombinedExp[i].y + (denseMultExp[j][i].y * w[j]);
            denseCombinedExp[i].z = denseCombinedExp[i].z + (denseMultExp[j][i].z * w[j]);
        }
    }
  
    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("C:/Users/stefa/Desktop/Capstone/repo/Facial-Tracking/data/face.obj"); //easy3D
 
    vector<easy3d::vec3> faceVerts;
    faceVerts.reserve(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        float x = denseCombinedExp[i].x;
        float y = denseCombinedExp[i].y;
        float z = denseCombinedExp[i].z;

        faceVerts.push_back(easy3d::vec3(x, y, z));
    }
    easy3d::logging::initialize();

    //-------------------------------------------------------------

    // Create the default Easy3D viewer.
    // Note: a viewer must be created before creating any drawables.
    easy3d::Viewer viewer("3d visualization");

    auto surface = new easy3d::TrianglesDrawable("faces");
    // Upload the vertex positions of the surface to the GPU.
    surface->update_vertex_buffer(faceVerts);
    // Upload the vertex indices of the surface to the GPU.
    surface->update_element_buffer(meshIndices);
    // Add the drawable to the viewer
    viewer.add_drawable(surface);

    //-------------------------------------------------------------
    // Create a PointsDrawable to visualize the vertices of the "bunny".
    // Only the vertex positions have to be sent to the GPU for visualization.
    auto vertices = new easy3d::PointsDrawable("vertices");
    // Upload the vertex positions to the GPU.
    vertices->update_vertex_buffer(faceVerts);
    // Set a color for the vertices (here we want a red color).
    vertices->set_uniform_coloring(easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a
    // Three options are available for visualizing points:
    //      - PLAIN: plain points (i.e., each point is a square on the screen);
    //      - SPHERE: each point is visualized a sphere;
    //      - SURFEL: each point is visualized an oriented disk.
    // In this example, let's render the vertices as spheres.
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    // Set the vertices size (here 10 pixels).
    vertices->set_point_size(10);
    // Add the drawable to the viewer
    viewer.add_drawable(surface);
    //viewer.add_drawable(vertices);


    viewer.fit_screen();
    // Run the viewer
    viewer.run();





    

//    // Convert Euler angles to rotation matrix
//    cv::Mat R;
//    cv::Rodrigues(rvec, R);
//
//    // Combine 3x3 rotation and 3x1 translation into 4x4 transformation matrix
//    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
//    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;
//    T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;
//
//    // Transform object
//    std::vector<cv::Mat> cameraVec;
//    for (auto& vec : combinedExp) {
//        double data[4] = { vec.x, vec.y, vec.z, 1 };
//        cv::Mat vector4d = cv::Mat(4, 1, CV_64F, data);
//        cv::Mat result = T * vector4d;
//        cameraVec.push_back(result);
//    }
//
//    // Project points onto image
//
//    std::vector<cv::Point2f> imageVec;
//    for (auto& vec : cameraVec) {
//        cv::Point2f result;
//        result.x = f * vec.at<double>(0, 0) / vec.at<double>(2, 0) + cx;
//        result.y = f * vec.at<double>(1, 0) / vec.at<double>(2, 0) + cy;
//
//        imageVec.push_back(result);
//    }
//    //cout << T << endl << cameraVec[0] << endl;
//    cv::projectPoints(combinedExp, rvec, tvec, cameraMatrix, cv::Mat(), imageVec);
//
//
//    cv::Mat visualImage = image.clone();
//    double sc = 1;
//    cv::resize(visualImage, visualImage, cv::Size(visualImage.cols * sc, visualImage.rows * sc));
//    for (int i = 0; i < imageVec.size(); i++) {
//        //cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 255, 0), 1);
////        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);
//
//        cv::circle(visualImage, imageVec[i] * sc, 1, cv::Scalar(0, 0, 255), sc);             // 3d projections (red)
//        cv::circle(visualImage, lmsVec[i] * sc, 1, cv::Scalar(0, 255, 0), sc);               // 2d landmarks   (green)
//
////        cv::putText(visualImage, std::to_string(i), lmsVec[i] * sc, 3, 0.4, cv::Scalar::all(255), 1);
//
//    }
    //cv::imshow("visualImage", visualImage);
    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);

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
