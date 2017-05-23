#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
#include <iostream>
#include <functional>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

enum SensorType {
  LASER,
  RADAR
};

// function pointer for measurement function
typedef std::function<std::tuple<VectorXd, MatrixXd> (float dt, const VectorXd &x)> ModelFunction;
typedef std::function<MatrixXd (float dt, const VectorXd &x)> ProcessNoiseFunction;

typedef std::function<VectorXd (const VectorXd &x)> SensorFunc;
typedef std::function<MatrixXd (const VectorXd &x)> SensorJacobianFunc;

typedef std::tuple<ModelFunction, ProcessNoiseFunction> DynamicModel;
typedef std::tuple<MatrixXd, SensorFunc, SensorJacobianFunc> SensorModel;

inline SensorModel MakeLinearSensor(const MatrixXd &R, const MatrixXd &H)
{
  auto measurementFunc = SensorFunc([H](const VectorXd& x){return H*x;});
  auto jacobianFunc = SensorJacobianFunc([H](const VectorXd& x){return H;});
  return std::make_tuple(R, measurementFunc, jacobianFunc);
}

class KalmanFilter {
public:

  // state vector
  VectorXd x_;

  // state covariance matrix
  MatrixXd P_;

  // identity matrix
  MatrixXd I_;

  DynamicModel dynamicModel_;
  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(const VectorXd &x_in, const MatrixXd &P_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict(float delta_T);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const VectorXd &z, const SensorModel& sensor);

  /**
   * Updates the state by using the given measurement and predicted measurement
   * @param z The measurement at k+1
   * @param z_pred The predicted measurement at k+1
   * @param H The jacobian of the measurement function
   */
  void UpdateWithPrediction(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &H, const MatrixXd &R);

private:
  DynamicModel getDynamicModel();
  ProcessNoiseFunction getProcessNoiseFunction();
  ModelFunction getModelFunction();

};

#endif /* KALMAN_FILTER_H_ */
