#include "FusionEKF.h"
//#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;


  // START HERE
  /* Sensor models */
  // Laser Sensor covariance and model matrix
  const MatrixXd H_laser = (MatrixXd(2, 4) << 1, 0, 0, 0,
      0, 1, 0, 0).finished();
  const MatrixXd R_laser = (MatrixXd(2, 2) << 0.0225, 0,
      0, 0.0225).finished();

  // Radar Sensor
  // Covariance
  const MatrixXd R_radar = (MatrixXd(3, 3) << 0.09, 0, 0,
      0, 0.09, 0,
      0, 0, 0.09).finished();

  AddLinearSensor(SensorType::LASER, R_laser, H_laser);
  AddSensor(SensorType::RADAR, R_radar, RadarMeasurement);

}

VectorXd FusionEKF::RadarMeasurement(const VectorXd &x) {
  VectorXd z_out(3);
  float px = x[0];
  float py = x[1];
  float vx = x[2];
  float vy = x[3];

  z_out << sqrt(px * px + py * py),
      atan2(py, px),
      (px * vx + py * vy) / sqrt(px * px + py * py);

  return z_out;
}

void FusionEKF::Init(const MeasurementPackage &measurement) {
  /****************************************************************************
   *  Initialization
   ****************************************************************************/
  long long timestamp = measurement.timestamp_;
  const MatrixXd P0 = (MatrixXd(4, 4) << 50, 0, 0, 0,
      0, 50, 0, 0,
      0, 0, 100, 0,
      0, 0, 0, 100).finished();

  ekf_.Init(computeX(measurement), P0);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    Init(measurement_pack);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, sensors_[measurement_pack.sensor_type_]);
  } else {

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  //dt - expressed in seconds
    ekf_.Predict(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, sensors_[measurement_pack.sensor_type_]);
  }
  previous_timestamp_ = measurement_pack.timestamp_;

}

/**
* Add new sensor definition.
*/
void FusionEKF::AddSensor(SensorType type, const Eigen::MatrixXd &R, const SensorFunc &h, const SensorJacobianFunc &H) {
  sensors_[type] = std::make_tuple(R, h, H);
}

/**
* Add new linear sensor definition.
*/
void FusionEKF::AddLinearSensor(SensorType type, const Eigen::MatrixXd &R, const MatrixXd &H) {
  sensors_[type] = MakeLinearSensor(R, H);
}

VectorXd FusionEKF::computeX(const MeasurementPackage &measurement) {
  VectorXd x_0 = VectorXd(4);
  float rho;
  if (measurement.sensor_type_ == SensorType::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    float phi, rhodot;
    rho = measurement.raw_measurements_[0];
    phi = measurement.raw_measurements_[1];
    x_0 << rho * cos(phi), rho * sin(phi), 0, 0;
  } else if (measurement.sensor_type_ == SensorType::LASER) {
    /**
    Initialize state.
    */
    float px, py;
    px = measurement.raw_measurements_[0];
    py = measurement.raw_measurements_[1];
    rho = sqrt(px * px + py * py);
    x_0 << px, py, 0, 0;
  }
  if (is_initialized_ == false && abs(rho) >= 1e-4) {
    is_initialized_ = true;
  }
  return x_0;
}