#include "treelocpp/geometry.h"

#include <algorithm>
#include <cmath>

namespace treelocpp {

Eigen::Matrix3d QuatToRotation(double qx, double qy, double qz, double qw) {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    return q.normalized().toRotationMatrix();
}

Eigen::Matrix4d PoseToTransform(const Pose& pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = QuatToRotation(pose.qx, pose.qy, pose.qz, pose.qw);
    T.block<3, 1>(0, 3) = Eigen::Vector3d(pose.x, pose.y, pose.z);
    return T;
}

double YawFromPose(const Pose& pose) {
    return std::atan2(2.0 * (pose.qw * pose.qz + pose.qx * pose.qy),
                      1.0 - 2.0 * (pose.qy * pose.qy + pose.qz * pose.qz));
}

Eigen::Matrix2d YawRotation(double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    return R;
}

double WrapAngle(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

double RotationYaw(const Eigen::Matrix2d& R) {
    return std::atan2(R(1, 0), R(0, 0));
}

void EulerZYX(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw) {
    yaw = std::atan2(R(1, 0), R(0, 0));
    pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    if (std::abs(std::cos(pitch)) < 1e-8) {
        roll = 0.0;
    } else {
        roll = std::atan2(R(2, 1), R(2, 2));
    }
}

Eigen::Vector3d TreeUp(const Tree& tree) {
    if (!tree.has_axis) return Eigen::Vector3d::UnitZ();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(tree.axis, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = U * V.transpose();
    if (R.determinant() < 0.0) {
        U.col(2) *= -1.0;
        R = U * V.transpose();
    }
    Eigen::Vector3d up = R.col(2);
    if (up.z() < 0.0) up = -up;
    return up.normalized();
}

Tree TransformTree2D(const Tree& tree, const Eigen::Matrix2d& R, const Eigen::Vector2d& t) {
    Tree out = tree;
    const Eigen::Vector2d p = R * Eigen::Vector2d(tree.x, tree.y) + t;
    out.x = p.x();
    out.y = p.y();
    return out;
}

Eigen::MatrixXd LocalToLocal2D(const std::vector<Eigen::Vector2d>& pts,
                               const Pose& src,
                               const Pose& dst) {
    const Eigen::Matrix2d Rs = YawRotation(YawFromPose(src));
    const Eigen::Matrix2d Rd = YawRotation(YawFromPose(dst));
    const Eigen::Vector2d ts(src.x, src.y);
    const Eigen::Vector2d td(dst.x, dst.y);
    const Eigen::Matrix2d R = Rd.transpose() * Rs;
    const Eigen::Vector2d t = Rd.transpose() * (ts - td);
    Eigen::MatrixXd out(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); ++i) out.row(i) = (R * pts[i] + t).transpose();
    return out;
}

double PoseDistanceXY(const Pose& a, const Pose& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace treelocpp
