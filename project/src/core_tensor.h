#ifndef DEMO_CORE_TENSOR_H
#define DEMO_CORE_TENSOR_H

#include <iostream>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using Eigen::Vector3f;

typedef Eigen::Tensor<Eigen::Vector3f, 3> tensor3;

void buildCoreTensor(string& warehousePath, string& outfile, tensor3& coreTensor);
void writeTensor(const string& filename, tensor3& tensor);
void loadCoreTensor(const string& filename, tensor3& tensor);
void displayEntireTensor(tensor3& coreTensor);

#endif //DEMO_CORE_TENSOR_H

