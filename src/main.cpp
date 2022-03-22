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

    // vector representing 150 identity weights
    int numIdentities = 150;
    Eigen::VectorXf identity_w(numIdentities);
    float weightVal = 1.0 / 150.0;

    for (int i = 0; i < 150; i++)
    {
        identity_w[i] = weightVal;
    }


    int n_vectors = 73;
    std::vector<cv::Point3f> singleIdn(n_vectors);
    std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
    // 150 identities
    for (int j = 0; j < numIdentities; j++)
    {
        // 73 vertices
        for (int i = 0; i < 73; i++)
        {
            Eigen::Vector3f tens_vec = shapeTensor(j, 22, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            singleIdn[i] = conv_vec;
        }
        multIdn[j] = singleIdn;
    }


    // create an average face
    std::vector<cv::Point3f> combinedIdn(n_vectors); 
    for (int i = 0; i < 73; i++)
    {
        for (int j = 0; j < numIdentities; j++)
        {
            combinedIdn[i].x = combinedIdn[i].x + (multIdn[j][i].x * identity_w[j]);
            combinedIdn[i].y = combinedIdn[i].y + (multIdn[j][i].y * identity_w[j]);
            combinedIdn[i].z = combinedIdn[i].z + (multIdn[j][i].z * identity_w[j]);
        }
    }

    for (int i = 0; i < 73; i++)
    {
        cout << combinedIdn[i] << endl;
    }

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
    // create a matrix of every expression for identity 138
    
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

    // create new expression based on the weights
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

    // expression optimization
    optimize(lmsVec, poseVec, image, f, w);

    /********** DENSE EXPRESSION **********/

    int numDenseVerts = 11510;
    std::vector<cv::Point3f> denseSingleExp(numDenseVerts);
    std::vector<std::vector<cv::Point3f>> denseMultExp(numExpressions);
    // 47 expressions
    for (int j = 0; j < numExpressions; j++)
    {
        
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

    /********** DENSE EXPRESSION **********/

    /********** DENSE IDENTITY **********/


    std::vector<cv::Point3f> denseSingleIdn(numDenseVerts);
    std::vector<std::vector<cv::Point3f>> denseMultIdn(numIdentities);
   
    for (int j = 0; j < numIdentities; j++)
    {
        for (int i = 0; i < numDenseVerts; i++)
        {
            Eigen::Vector3f tens_vec = rawTensor(j, 0, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            denseSingleIdn[i] = conv_vec;
        }
        denseMultIdn[j] = denseSingleIdn;
    }

    std::vector<cv::Point3f> denseCombinedIdn(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numIdentities; j++)
        {
            denseCombinedIdn[i].x = denseCombinedIdn[i].x + (denseMultIdn[j][i].x * identity_w[j]);
            denseCombinedIdn[i].y = denseCombinedIdn[i].y + (denseMultIdn[j][i].y * identity_w[j]);
            denseCombinedIdn[i].z = denseCombinedIdn[i].z + (denseMultIdn[j][i].z * identity_w[j]);
        }
    }

    /********** DENSE IDENTITY **********/

    /********** 3D VISUALIZATION **********/
  
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

    vector<easy3d::vec3> faceVerts2;
    faceVerts2.reserve(numDenseVerts);

    for (int i = 0; i < numDenseVerts; i++)
    {
        float x = denseCombinedIdn[i].x;
        float y = denseCombinedIdn[i].y;
        float z = denseCombinedIdn[i].z;

        faceVerts2.push_back(easy3d::vec3(x, y, z));
    }

    vector<int> v0 = {5154, 11283, 5157, 11284, 5181, 11295, 5494, 10957, 4499, 10822, 4211, 10803, 4128, 10804};
    vector<int> v1 = {5043, 1089, 5172, 1151, 5170, 1316, 5499, 821, 4506, 685, 4151, 646, 4184, 664};
    vector<int> v2 = {1054,5164,1147,5162,1318,5502,822, 4220,686,4221,669,4194,668,4193};
    vector<int> v3 = {4973,11203,5168,11288,5503,10620,3845,10621,3875,10629,3857,10565,388,10545};
    vector<int> v4 = {1126,5117,1154,5178,1323,3855,497,3879,510,3861,501,3699,424,3701};
    vector<int> v5 = {10614,3883,10633,3865,10539,3613,10536,3831,10527,3716,10524,3806,10550,3620};
    vector<int> v6 = {10623, 3884, 10635, 3869,10573,3658,10574,3812,10575,3660,10576,3808,10578,3651};
    vector<int> v7 = {9190,3887,9180,3872,9045,3611,9150,6703,9053,3607,9137,3787,9052,3619};
    vector<int> v8 = {9188,6773,9178,6755,9078,6541,9148,6698,9079,6558,9135,6672,9080,6546};
    vector<int> v9 = {6739,1923,6741,1937,6767,1928,6752,1854,6720,1912,6719,1855,6693,1899};
    vector<int> v10 = {7994,2579,8058,2748,8391,1922,6762,1935,6749,1926,6748,1849,6714,1910};
    vector<int> v11 = {7811,2480,7852,2573,8046,2744,6806,1957,6804,1956,6801,1955,6802,6794}; //1951
    vector<int> v12 = {2461,7819,2484,8048,2574,8050,2472,7392,2244,7390,2108,7109,1990,6878};
    vector<int> v13 = {2548,7989,2549,7992,2550,7993,2578,8057,2740,8373,2245,7394,2109,6877};
    vector<int> v14 = {6001,8634,5997,8628,5732,8624,5730,8625,5941,8770,7397,9380,7118,9248};
    
    vector<int> allVerts = {
        5154, 11283, 5157, 11284, 5181, 11295, 5494, 10957, 4499, 10822, 4211, 10803, 4128, 10804,
        5043, 1089, 5172, 1151, 5170, 1316, 5499, 821, 4506, 685, 4151, 646, 4184, 664,
        1054,5164,1147,5162,1318,5502,822, 4220,686,4221,669,4194,668,4193,
        4973,11203,5168,11288,5503,10620,3845,10621,3875,10629,3857,10565,388,10545,
        1126,5117,1154,5178,1323,3855,497,3879,510,3861,501,3699,424,3701,
        10614,3883,10633,3865,10539,3613,10536,3831,10527,3716,10524,3806,10550,3620,
        10623, 3884, 10635, 3869,10573,3658,10574,3812,10575,3660,10576,3808,10578,3651,
        9190,3887,9180,3872,9045,3611,9150,6703,9053,3607,9137,3787,9052,3619,
        9188,6773,9178,6755,9078,6541,9148,6698,9079,6558,9135,6672,9080,6546,
        6739,1923,6741,1937,6767,1928,6752,1854,6720,1912,6719,1855,6693,1899,
        7994,2579,8058,2748,8391,1922,6762,1935,6749,1926,6748,1849,6714,1910,
        7811,2480,7852,2573,8046,2744,6806,1957,6804,1956,6801,1955,6802,6794,
        2461,7819,2484,8048,2574,8050,2472,7392,2244,7390,2108,7109,1990,6878,
        2548,7989,2549,7992,2550,7993,2578,8057,2740,8373,2245,7394,2109,6877,
        6001,8634,5997,8628,5732,8624,5730,8625,5941,8770,7397,9380,7118,9248
    };

    vector<easy3d::vec3> contourVerts(allVerts.size());
    for (int i = 0; i < contourVerts.size(); i++)
    {
        contourVerts[i] = faceVerts2[allVerts[i]];
    }

    easy3d::logging::initialize();

    //-------------------------------------------------------------

    // Create the default Easy3D viewer.
    // Note: a viewer must be created before creating any drawables.
    easy3d::Viewer viewer("3d visualization");

    auto surface = new easy3d::TrianglesDrawable("faces");
    // Upload the vertex positions of the surface to the GPU.
    surface->update_vertex_buffer(faceVerts2);
    // Upload the vertex indices of the surface to the GPU.
    surface->update_element_buffer(meshIndices);
    // Add the drawable to the viewer
    viewer.add_drawable(surface);

    auto vertices = new easy3d::PointsDrawable("vertices");
    // Upload the vertex positions to the GPU.
    vertices->update_vertex_buffer(contourVerts);
    // Set a color for the vertices (here we want a red color).
    vertices->set_uniform_coloring(easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a
   
    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    // Set the vertices size (here 10 pixels).
    vertices->set_point_size(10);
    // Add the drawable to the viewer
    viewer.add_drawable(surface);
   // viewer.add_drawable(vertices);

    viewer.fit_screen();
    // Run the viewer
    viewer.run();

    /********** 3D VISUALIZATION **********/



    

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
