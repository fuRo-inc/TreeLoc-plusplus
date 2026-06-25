#pragma once

#include <vector>

#include <Eigen/Dense>

#include "treelocpp/types.h"

namespace treelocpp {

Eigen::Matrix3d QuatToRotation(double qx, double qy, double qz, double qw);
Eigen::Matrix4d PoseToTransform(const Pose& pose);
double YawFromPose(const Pose& pose);
Eigen::Matrix2d YawRotation(double yaw);
double WrapAngle(double angle);
double RotationYaw(const Eigen::Matrix2d& R);
void EulerZYX(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw);
Eigen::Vector3d TreeUp(const Tree& tree);
Tree TransformTree2D(const Tree& tree, const Eigen::Matrix2d& R, const Eigen::Vector2d& t);
Eigen::MatrixXd LocalToLocal2D(const std::vector<Eigen::Vector2d>& pts,
                               const Pose& src,
                               const Pose& dst);
double PoseDistanceXY(const Pose& a, const Pose& b);

}  // namespace treelocpp
