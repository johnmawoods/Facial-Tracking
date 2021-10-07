#include <iostream>
#include <filesystem>

#include "core_tensor.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

string WAREHOUSE_PATH = "data/FaceWarehouse/";
string CORE_TENSOR_PATH = "data/core_tensor.bin";

int main() {
    // Core tensor: 150 users X 47 expressions X 11510 vertices
    tensor3 coreTensor(150, 47, 11510);

    // If core tensor file does not exist, build it from face warehouse
    if (std::filesystem::exists(CORE_TENSOR_PATH)) {
        loadCoreTensor(CORE_TENSOR_PATH, coreTensor);
    }
    else {
        buildCoreTensor(WAREHOUSE_PATH, CORE_TENSOR_PATH, coreTensor);
    }

    // displayEntireTensor(coreTensor);
}