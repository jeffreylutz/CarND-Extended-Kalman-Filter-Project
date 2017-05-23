#ifndef FusionEKF_H_
#define FusionEKF_H_

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <memory>

#include "measurement_package.h"
#include "kalman_filter.h"

typedef std::unordered_map<SensorType, SensorModel, EnumClassHash> SensorMap;

class FusionEKF {
public:
  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  void Init(const MeasurementPackage& measurement);
  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Add sensor definition to FusionEKF
   * @param type
   * @param measCov
   * @param measModel
   * @param measJac
   */
  void AddSensor(SensorType type, const Eigen::MatrixXd& measCov, const SensorFunc& measModel, const SensorJacobianFunc& measJac = NULL);

  /**
   * Add linear model
   * @param type
   * @param measCov
   * @param measMat
   */
  void AddLinearSensor(SensorType type, const Eigen::MatrixXd& measCov, const MatrixXd& measMat);
  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

  // check whether the tracking toolbox was initiallized or not (first measurement)
  bool is_initialized_;

  static VectorXd RadarMeasurement(const VectorXd &x);

private:
  // previous timestamp
  long long previous_timestamp_;

  // Map containing sensor definitions
  SensorMap sensors_;
  VectorXd computeX(const MeasurementPackage &measurement);
};

#endif /* FusionEKF_H_ */
