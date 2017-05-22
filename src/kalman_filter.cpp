#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in, const MatrixXd &P_in) {
  x_ = x_in;
  P_ = P_in;
  dynamicModel_ = getDynamicModel();
  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict(float delta_T) {
  // KF Prediction step
  ProcessNoiseFunction noiseFunction;
  ModelFunction dynamicFunc;
  MatrixXd F;

  std::tie(dynamicFunc, noiseFunction) = dynamicModel_;
  std::tie(x_, F) = dynamicFunc(delta_T, x_);

  auto Q_ = noiseFunction(delta_T, x_);
  P_ = F * P_ * F.transpose() + Q_;
}

void KalmanFilter::UpdateEKF(const Eigen::VectorXd &z, const SensorModel &sensor) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  SensorFunc h_func;
  SensorJacobianFunc H_func;
  MatrixXd R, H;
  VectorXd z_pred;

  std::tie(R, h_func, H_func) = sensor;

  z_pred = h_func(x_);
  if (H_func == NULL) {
    H = Tools::CalculateJacobian(h_func, x_);
  } else {
    H = H_func(x_);
  }

  UpdateWithPrediction(z, z_pred, H, R);
}

void KalmanFilter::UpdateWithPrediction(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &H, const MatrixXd &R) {
  MatrixXd Ht = H.transpose();
  VectorXd y = z - z_pred;
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H) * P_;
}

DynamicModel KalmanFilter::getDynamicModel() {
  return DynamicModel(getModelFunction(), getProcessNoiseFunction());
}

ProcessNoiseFunction KalmanFilter::getProcessNoiseFunction() {

  return ProcessNoiseFunction([](float dt, const VectorXd &x) mutable {
    // Process noise
    const float noise_ax = 7, noise_ay = 7;

    MatrixXd Q = MatrixXd(4, 4);
    float dt2, dt3, dt4;
    dt2 = dt*dt;
    dt3 = dt2*dt;
    dt4 = dt3*dt;

    Q << noise_ax*dt4/4, 0, noise_ax*dt3/2, 0,
        0, noise_ay*dt4/4, 0, noise_ay*dt3/2,
        noise_ax*dt3/2, 0, noise_ax*dt2,  0,
        0, noise_ay*dt3/2, 0, noise_ay*dt2;
    return Q;
  });
}

ModelFunction KalmanFilter::getModelFunction() {
  return ModelFunction([](float dt, const VectorXd &x){
    MatrixXd F = MatrixXd(4,4);
    F << 1, 0, dt, 0,
        0, 1,  0, dt,
        0, 0,  1, 0,
        0, 0,  0, 1;
    return std::tuple<VectorXd, MatrixXd>(F*x, F);
  });
}