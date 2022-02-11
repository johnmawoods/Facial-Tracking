#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <iostream>
#include <vector>
#include <random>

#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;


vector<double> optimize(const Eigen::VectorXf& w, const Eigen::VectorXf& lms,
    const Eigen::VectorXf& pose, const cv::Mat& image, float f);

#endif //OPTIMIZATION_H