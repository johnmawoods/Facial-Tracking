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

void buildCoreTensor(string& warehousePath, string& outfile, tensor3& coreTensor);
void writeTensor(const string& filename, tensor3& tensor);
void loadCoreTensor(const string& filename, tensor3& tensor);
void displayEntireTensor(tensor3& coreTensor);
vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image);
#endif //DEMO_CORE_TENSOR_H

