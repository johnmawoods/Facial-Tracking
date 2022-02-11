#include <iostream>
#include <vector>
#include <random>

#include "../include/optimization.h"

#include "ceres/ceres.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;

struct ReprojectErrorExp {

	ReprojectErrorExp(int numObservations, const vector<cv::Point2f>& x, const vector<cv::Point2f>& y) {
        _numObservations = numObservations;
        _x.resize(numObservations);
        _y.resize(numObservations);
        std::copy(x.begin(), x.end(), _x.begin());
        std::copy(y.begin(), y.end(), _y.begin());
    }

    template <typename T>
    bool operator()(const T* w, T* residual) const {

        // 73 true landmarks and 73 estimations
        // 47 weights representing the 47 expressions
        for (int i = 0; i < _numObservations; i++)
            //residual[i] = T(_y[i]) - (w[0] * (T(_x[i]) * T(_x[i])) + (w[1] * T(_x[i])) + w[2]);

        return true;
    }

private:
    int                 _numObservations = 0;
    vector<cv::Point2f>      _x;
    vector<cv::Point2f>      _y;
};


vector<double> optimize(const Eigen::VectorXf& w, const Eigen::VectorXf& lms,
    const Eigen::VectorXf& pose, const cv::Mat& image, float f)
{
	int numExpressions = 47;
	int numLms = lms.size();
	float cx = image.cols / 2.0;
	float cy = image.rows / 2.0;

	vector<double> w(numExpressions, 0);        // numExpressions = 47
	ceres::Problem problem;

	vector<cv::Point2f> gtLms;
	gtLms.reserve(numLms);
	for (int i = 0; i < numLms; i++) {
		float gtX = (lms[i].x - cx) / f;
		float gtY = (lms[i].y - cy) / f;
		gtLms.emplace_back(gtX, gtY);
	}

	ReprojectErrorExp* repErrFunc = new ReprojectErrorExp(pose, numLms, sparseBlendshapes, gtLms);   // upload the required parameters
	ceres::CostFunction* optimTerm = new ceres::AutoDiffCostFunction<ReprojectErrorExp, ceres::DYNAMIC, 46>(repErrFunc, numLms * 2);  // times 2 becase we have gtx and gty
	problem.AddResidualBlock(optimTerm, NULL, &w[0]);


	// for (int i = 0; i < _numExpressions - 1; i++) {
		// problem.SetParameterLowerBound(&w[0], i, 0.0);   // first argument must be w of ZERO and the second is the index of interest
		// problem.SetParameterUpperBound(&w[0], i, 1.0);    // also the boundaries should be set after adding the residual block
	// }

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	cout << summary.BriefReport() << endl << endl;
	for (int i = 0; i < _numExpressions; i++)
		w_exp(i) = w[i];


	return summary.termination_type == ceres::TerminationType::CONVERGENCE;

}