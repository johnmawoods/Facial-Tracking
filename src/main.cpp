#include <iostream>
#include <filesystem>
#include <vector>
#include <random>

#include "ceres/ceres.h"

#include "../include/tensor.h"
#include "../include/optimization.h"
#include "../include/identityOptimization.h"

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

vector<int> v0 = { 5154, 11283, 5157, 11284, 5181, 11295, 5494, 10957, 4499, 10822, 4211, 10803, 4128, 10804 };
vector<int> v1 = { 5043, 1089, 5172, 1151, 5170, 1316, 5499, 821, 4506, 685, 4151, 646, 4184, 664 };
vector<int> v2 = { 1054,5164,1147,5162,1318,5502,822, 4220,686,4221,669,4194,668,4193 };
vector<int> v3 = { 4973,11203,5168,11288,5503,10620,3845,10621,3875,10629,3857,10565,388,10545 };
vector<int> v4 = { 1126,5117,1154,5178,1323,3855,497,3879,510,3861,501,3699,424,3701 };
vector<int> v5 = { 10614,3883,10633,3865,10539,3613,10536,3831,10527,3716,10524,3806,10550,3620 };
vector<int> v6 = { 10623, 3884, 10635, 3869,10573,3658,10574,3812,10575,3660,10576,3808,10578,3651 };
vector<int> v7 = { 9190,3887,9180,3872,9045,3611,9150,6703,9053,3607,9137,3787,9052,3619 };
vector<int> v8 = { 9188,6773,9178,6755,9078,6541,9148,6698,9079,6558,9135,6672,9080,6546 };
vector<int> v9 = { 6739,1923,6741,1937,6767,1928,6752,1854,6720,1912,6719,1855,6693,1899 };
vector<int> v10 = { 7994,2579,8058,2748,8391,1922,6762,1935,6749,1926,6748,1849,6714,1910 };
vector<int> v11 = { 7811,2480,7852,2573,8046,2744,6806,1957,6804,1956,6801,1955,6802,6794 }; //1951
vector<int> v12 = { 2461,7819,2484,8048,2574,8050,2472,7392,2244,7390,2108,7109,1990,6878 };
vector<int> v13 = { 2548,7989,2549,7992,2550,7993,2578,8057,2740,8373,2245,7394,2109,6877 };
vector<int> v14 = { 6001,8634,5997,8628,5732,8624,5730,8625,5941,8770,7397,9380,7118,9248 };


int main() {
    void createAllExpressions(tensor3 tensor, Eigen::VectorXf identity_w, int numVerts, std::vector<std::vector<cv::Point3f>>& avgMultExp);
    void createAllIdentities(tensor3 tensor, Eigen::VectorXf w, int numVerts, std::vector<std::vector<cv::Point3f>>& allIdnOptExp);
    void linearCombination(int numVerts, int numCombinations, std::vector<std::vector<cv::Point3f>> mult, Eigen::VectorXf w, std::vector<cv::Point3f>&linCombo);
    void visualization3D(int numVerts, std::vector<cv::Point3f> linCombo);
    void getPose(std::vector<float>&poseVec, const cv::Mat & rvec, const cv::Mat & tvec);

    /* CREATE RAW AND SHAPE TENSOR */
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
    /* CREATE RAW AND SHAPE TENSOR */


    /* VARIABLES */
    int numIdentities = 150;
    int numExpressions = 47;
    int numShapeVerts = 73;
    int numDenseVerts = 11510;

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
    Eigen::VectorXf w(numExpressions);
    for (int i = 0; i < numExpressions; i++)
    {
        w[i] = 0;
    }
    w[0] = 1;

    // vector representing 150 identity weights
    Eigen::VectorXf identity_w(numIdentities);
    float weightVal = 1.0 / 150.0;
    for (int i = 0; i < 150; i++)
    {
        identity_w[i] = weightVal;
    }
    /* VARIABLES */

    cout << "1" << endl;

    /* creates a matrix of all the expressions for the average identity */
    std::vector<std::vector<cv::Point3f>> avgMultExp(numExpressions);
    createAllExpressions(shapeTensor, identity_w, numShapeVerts, avgMultExp);
    std::vector<cv::Point3f> singleFace(numShapeVerts);
    for (int i = 0; i < numShapeVerts; i++)
    {
        singleFace[i].x = avgMultExp[0][i].x;
        singleFace[i].y = avgMultExp[0][i].y;
        singleFace[i].z = avgMultExp[0][i].z;
    }
    
    cout << "2" << endl;

    /* creates vector of average identity with neutral expression */
    /* we use this for initial pose estimation */
    std::vector<cv::Point3f> singleIdn(numShapeVerts);
    std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
 
    for (int j = 0; j < numIdentities; j++)
    {
        for (int i = 0; i < numShapeVerts; i++)
        {
            Eigen::Vector3f tens_vec = shapeTensor(j, 0, i);
            cv::Point3f conv_vec;
            conv_vec.x = tens_vec.x();
            conv_vec.y = tens_vec.y();
            conv_vec.z = tens_vec.z();
            singleIdn[i] = conv_vec;
        }
        multIdn[j] = singleIdn;
    }

    // create an average face
    std::vector<cv::Point3f> combinedIdn(numShapeVerts);
    linearCombination(numShapeVerts, numIdentities, multIdn, identity_w, combinedIdn);
   
    cout << "3" << endl;
    // pose estimation
    // solves rvec and tvec for average face neutral expression
    cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);

    std::vector<float> poseVec(6, 0);
    getPose(poseVec, rvec, tvec);
    
    cout << "44" << endl;
    // expression optimization
    optimize(lmsVec, poseVec, image, f, w, avgMultExp);

    cout << "4" << endl;
    /* apply optimized expression weights and create a vector of every identity */
    std::vector<std::vector<cv::Point3f>> allIdnOptExp(numIdentities);
    createAllIdentities(shapeTensor, w, 73, allIdnOptExp);
    linearCombination(numShapeVerts, numIdentities, allIdnOptExp, identity_w, combinedIdn);
    cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
    getPose(poseVec, rvec, tvec);
    cout << "5" << endl;

    /* optimize for identity */
    identityOptimize(lmsVec, poseVec, image, f, identity_w, allIdnOptExp);
    cout << "6" << endl;
    /* create new face based on optimized w_exp and w_idn for pose estimation */
    createAllExpressions(shapeTensor, identity_w, numShapeVerts, avgMultExp);
    linearCombination(numShapeVerts, numExpressions, avgMultExp, w, combinedIdn);
    
    cout << "7" << endl;
    cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
    getPose(poseVec, rvec, tvec);

    // expression optimization
    optimize(lmsVec, poseVec, image, f, w, avgMultExp);
    cout << "8" << endl;
    // create new face based on optimized expression weights
    createAllIdentities(shapeTensor, w, 73, allIdnOptExp);
    linearCombination(numShapeVerts, numIdentities, allIdnOptExp, identity_w, combinedIdn);
    cv::solvePnP(combinedIdn, lmsVec, cameraMatrix, cv::Mat(), rvec, tvec);
    getPose(poseVec, rvec, tvec);
    cout << "9" << endl;
    /* optimize for identity */
    identityOptimize(lmsVec, poseVec, image, f, identity_w, allIdnOptExp);
    cout << "10" << endl;
    // expression and identity weights have now both been optimized


    /********** OPTIMIZED DENSE FACE **********/

    std::vector<cv::Point3f> denseCombinedIdn(numDenseVerts);
    createAllExpressions(shapeTensor, identity_w, numDenseVerts, avgMultExp);
    cout << "line = " << __LINE__ << endl;
    for (int i = 0; i < numDenseVerts; i++)
    {
        for (int j = 0; j < numExpressions; j++)
        {
            denseCombinedIdn[i].x += (avgMultExp[j][i].x * w[j]);
            denseCombinedIdn[i].y += (avgMultExp[j][i].y * w[j]);
            denseCombinedIdn[i].z += (avgMultExp[j][i].z * w[j]);
        }
    }
    cout << "11" << endl;
    /********** OPTIMIZED DENSE FACE **********/



    int key = cv::waitKey(0) % 256;
    if (key == 27)                        // Esc button is pressed
        exit(1);

    return 0;


}

void visualization3D(int numVerts, std::vector<cv::Point3f> linCombo)
{
    vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("C:/Users/stefa/Desktop/Capstone/repo/Facial-Tracking/data/face.obj");

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
   vector<easy3d::vec3> faceVerts;
   faceVerts.reserve(numVerts);

   for (int i = 0; i < numVerts; i++)
   {
       float x = linCombo[i].x;
       float y = linCombo[i].y;
       float z = linCombo[i].z;

       faceVerts.push_back(easy3d::vec3(x, y, z));
   }
   /*vector<easy3d::vec3> contourVerts(allVerts.size());
   for (int i = 0; i < contourVerts.size(); i++)
   {
       contourVerts[i] = faceVerts[allVerts[i]];
   }*/
    easy3d::logging::initialize();

    // Create the default Easy3D viewer.
    // Note: a viewer must be created before creating any drawables.
    easy3d::Viewer viewer("3d visualization");

    auto surface = new easy3d::TrianglesDrawable("faces");
    surface->update_vertex_buffer(faceVerts);
    surface->update_element_buffer(meshIndices);
    viewer.add_drawable(surface);

    auto vertices = new easy3d::PointsDrawable("vertices");
    vertices->update_vertex_buffer(faceVerts);
    vertices->set_uniform_coloring(easy3d::vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a

    vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
    vertices->set_point_size(10);
    //viewer.add_drawable(vertices);

    viewer.fit_screen();
    viewer.run();
}

 void linearCombination(int numVerts, int numCombinations, std::vector<std::vector<cv::Point3f>> mult, Eigen::VectorXf w, std::vector<cv::Point3f>& linCombo)
{
    for (int i = 0; i < numVerts; i++)
    {
        for (int j = 0; j < numCombinations; j++)
        {
            linCombo[i].x += (mult[j][i].x * w[j]);
            linCombo[i].y += (mult[j][i].y * w[j]);
            linCombo[i].z += (mult[j][i].z * w[j]);
        }
    }
}

/* creates a matrix of all the expressions for the given identity weights */
void createAllExpressions(tensor3 tensor, 
    Eigen::VectorXf identity_w, int numVerts, std::vector<std::vector<cv::Point3f>>& avgMultExp) {
    /* creates a matrix of all the expressions for the average identity */
    int numExpressions = 47;
    int numIdentities = 150;
    for (int e = 0; e < numExpressions; e++)
    {
        std::vector<cv::Point3f> singleIdn(numVerts);
        std::vector<std::vector<cv::Point3f>> multIdn(numIdentities);
        // 150 identities
        for (int j = 0; j < numIdentities; j++)
        {
            for (int i = 0; i < numVerts; i++)
            {
                Eigen::Vector3f tens_vec = tensor(j, e, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleIdn[i] = conv_vec;
            }
            multIdn[j] = singleIdn;
        }
        // create an average face
        std::vector<cv::Point3f> combinedIdn(numVerts);
        linearCombination(numVerts, numIdentities, multIdn, identity_w, combinedIdn);
        avgMultExp[e] = combinedIdn;
    }
}

/* apply optimized expression weights and create a vector of every identity */
/* vector length is 150 */
void createAllIdentities(tensor3 tensor, 
    Eigen::VectorXf w, int numVerts, std::vector<std::vector<cv::Point3f>>& allIdnOptExp) {
    int numExpressions = 47;
    int numIdentities = 150;

    for (int idnNum = 0; idnNum < numIdentities; idnNum++)
    {
        std::vector<cv::Point3f> singleExp(numVerts);
        std::vector<std::vector<cv::Point3f>> multExp(numExpressions);
        // 47 expressions
        for (int j = 0; j < numExpressions; j++)
        {
            for (int i = 0; i < numVerts; i++)
            {
                Eigen::Vector3f tens_vec = tensor(idnNum, j, i);
                cv::Point3f conv_vec;
                conv_vec.x = tens_vec.x();
                conv_vec.y = tens_vec.y();
                conv_vec.z = tens_vec.z();
                singleExp[i] = conv_vec;
            }
            multExp[j] = singleExp;
        }

        std::vector<cv::Point3f> combinedExp(numVerts);
        linearCombination(numVerts, numExpressions, multExp, w, combinedExp);
        allIdnOptExp[idnNum] = combinedExp;
    }
}

void getPose(std::vector<float>& poseVec, const cv::Mat& rvec, const cv::Mat& tvec)
{
    poseVec[0] = rvec.at<double>(0);
    poseVec[1] = rvec.at<double>(1);
    poseVec[2] = rvec.at<double>(2);

    poseVec[3] = tvec.at<double>(0);
    poseVec[4] = tvec.at<double>(1);
    poseVec[5] = tvec.at<double>(2);
    for (auto pose : poseVec)
    {
        cout << pose << endl;
    }
}



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
