#ifndef DEMO_CORE_TENSOR_H
#define DEMO_CORE_TENSOR_H

#include "utility.h"
#include <iostream>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using Eigen::Vector3f;

namespace asu {

    class Tensor : public Utility {

    public:
        Tensor() {}
        ~Tensor() {}

    private:

    };

}

typedef Eigen::Tensor<Eigen::Vector3f, 3> tensor3;

void buildRawTensor(string& warehousePath, string& outfile, tensor3& rawTensor);
void writeTensor(const string& filename, tensor3& tensor);
void loadRawTensor(const string& filename, tensor3& tensor);
void displayEntireTensor(tensor3& rawTensor);

void loadShapeTensor(string& SHAPE_TENSOR_PATH, tensor3& shapeTensor);
void buildShapeTensor(tensor3& rawTensor, string& outfile, tensor3& shapeTensor);
void writeShapeTensor(const string& filename, tensor3& tensor);

vector<uint32_t> readMeshTriangleIndicesFromFile(const std::string& path);

vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image);

#endif //DEMO_CORE_TENSOR_H

