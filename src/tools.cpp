#include <iostream>
#include "tools.h"

VectorXd Tools::CalculateRMSE(const vector <VectorXd> &estimations,
                              const vector <VectorXd> &ground_truth) {
  /**
* A helper method to calculate RMSE.
*/
  VectorXd rmse(4), dx(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() == 0) {
    cerr << "Need at least one estimate, boss!" << endl;
    return rmse;
  } else if (estimations.size() != ground_truth.size()) {
    cerr << "estimations.size() != ground_truth.size()" << endl;
    return rmse;
  }
  //  * the estimation vector size should equal ground truth vector size
  //accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    // ... your code here
    dx = (estimations[i] - ground_truth[i]);
    rmse += dx.cwiseAbs2();
  }

  //calculate the mean
  rmse /= estimations.size();

  //calculate the squared root
  rmse = rmse.cwiseSqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  // Calculate a Jacobian here.
  MatrixXd Hj_(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  if (fabs(px) < MIN_DIVISOR and fabs(py) < MIN_DIVISOR) {
    px = MIN_DIVISOR;
    py = MIN_DIVISOR;
  }

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);


  //compute the Jacobian matrix
  Hj_ << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj_;
}

