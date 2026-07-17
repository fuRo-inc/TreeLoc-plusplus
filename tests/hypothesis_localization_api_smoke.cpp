#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "treelocpp/config.h"
#include "treelocpp/descriptors.h"
#include "treelocpp/localize_hypotheses.h"
#include "treelocpp/types.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void RequireClose(double actual,
                  double expected,
                  double tolerance,
                  const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

treelocpp::Pose MakePose(double x,
                         double y,
                         double z,
                         double yaw) {
    treelocpp::Pose pose;
    pose.x = x;
    pose.y = y;
    pose.z = z;
    pose.qz = std::sin(0.5 * yaw);
    pose.qw = std::cos(0.5 * yaw);
    return pose;
}

treelocpp::Tree MakeTree(double x,
                         double y,
                         double dbh,
                         bool partial) {
    treelocpp::Tree tree;
    tree.x = x;
    tree.y = y;
    tree.z = 0.0;
    tree.score = 1.0;
    tree.reconstructed = partial ? 0 : 1;
    tree.number_clusters = partial ? 1 : 3;
    if (!partial) {
        tree.dbh = dbh;
        tree.dbh_valid = true;
        tree.dbh_approximation = dbh;
    }
    return tree;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected TreeLoc++ config path");
        }

        treelocpp::Config config;
        std::string config_error;
        if (!treelocpp::LoadConfig(
                std::filesystem::path(argv[1]), config, &config_error)) {
            throw std::runtime_error(config_error);
        }
        config.neighbor_augment = false;
        treelocpp::RefreshDerivedConfig(config);

        const std::vector<treelocpp::Tree> map_trees = {
            MakeTree(0.4, 1.2, 0.18, false),
            MakeTree(2.8, -0.7, 0.26, false),
            MakeTree(-1.5, 3.4, 0.34, false),
            MakeTree(4.2, 2.1, 0.42, false),
            MakeTree(-3.1, -1.8, 0.50, false),
            MakeTree(1.1, 5.0, 0.58, false),
        };

        const double map_query_x = 3.0;
        const double map_query_y = -1.5;
        const double map_query_yaw = 20.0 * kPi / 180.0;
        const double c = std::cos(map_query_yaw);
        const double s = std::sin(map_query_yaw);

        std::vector<treelocpp::Tree> query_trees;
        query_trees.reserve(map_trees.size());
        for (const auto& map_tree : map_trees) {
            const double dx = map_tree.x - map_query_x;
            const double dy = map_tree.y - map_query_y;
            query_trees.push_back(MakeTree(
                c * dx + s * dy,
                -s * dx + c * dy,
                0.0,
                true));
        }

        const double odom_query_x = 5.0;
        const double odom_query_y = 2.0;
        const double odom_query_yaw = 10.0 * kPi / 180.0;
        treelocpp::Pose odom_query_pose = MakePose(
            odom_query_x, odom_query_y, 2.0, odom_query_yaw);
        odom_query_pose.stamp = 123.5;

        const treelocpp::FrameData query = treelocpp::BuildFrameData(
            42, odom_query_pose, query_trees, config);

        treelocpp::Dataset database;
        for (int index = 0; index < 3; ++index) {
            treelocpp::Pose map_anchor_pose = MakePose(
                0.0, 0.0, 0.0, 0.0);
            treelocpp::FrameData frame = treelocpp::BuildFrameData(
                index, map_anchor_pose, map_trees, config);
            database.frame_to_slot[index] = database.frames.size();
            database.trajectory.push_back(map_anchor_pose);
            database.frames.push_back(std::move(frame));
        }

        const treelocpp::Pose map_query_prior = MakePose(
            map_query_x, map_query_y, 4.25, map_query_yaw);

        treelocpp::HypothesisLocalizationOptions options;
        options.search_radius = 4.0;
        options.min_consensus_support = 3;

        const treelocpp::HypothesisLocalizationResult result =
            treelocpp::LocalizeHypotheses(
                query, database, map_query_prior, config, options);

        Require(result.ok, "localization was rejected");
        Require(!result.ambiguous, "localization was marked ambiguous");
        Require(result.status == "ok", "unexpected result status");
        Require(result.query_index == 42, "query index was not preserved");
        Require(result.candidate_count == 3, "unexpected candidate count");
        Require(result.support == 3, "unexpected consensus support");
        Require(result.runner_up_support == 0,
                "unexpected runner-up support");
        Require(result.map_index >= 0 && result.map_index < 3,
                "unexpected representative map index");
        Require(result.consensus_database_indices.size() == 3,
                "unexpected consensus member count");
        Require(result.pairs == 6, "unexpected matched pair count");

        const double estimated_map_yaw = std::atan2(
            result.T_map_query(1, 0), result.T_map_query(0, 0));
        RequireClose(result.T_map_query(0, 3), map_query_x, 1.0e-5,
                     "map query x mismatch");
        RequireClose(result.T_map_query(1, 3), map_query_y, 1.0e-5,
                     "map query y mismatch");
        RequireClose(result.T_map_query(2, 3), 4.25, 1.0e-12,
                     "map query z must remain the prior z");
        RequireClose(estimated_map_yaw, map_query_yaw, 1.0e-5,
                     "map query yaw mismatch");

        const double expected_map_odom_yaw =
            map_query_yaw - odom_query_yaw;
        const double cy = std::cos(expected_map_odom_yaw);
        const double sy = std::sin(expected_map_odom_yaw);
        const double expected_map_odom_x =
            map_query_x - (cy * odom_query_x - sy * odom_query_y);
        const double expected_map_odom_y =
            map_query_y - (sy * odom_query_x + cy * odom_query_y);
        const double estimated_map_odom_yaw = std::atan2(
            result.T_map_odom(1, 0), result.T_map_odom(0, 0));

        RequireClose(result.T_map_odom(0, 3), expected_map_odom_x, 1.0e-5,
                     "map-to-odom x mismatch");
        RequireClose(result.T_map_odom(1, 3), expected_map_odom_y, 1.0e-5,
                     "map-to-odom y mismatch");
        RequireClose(result.T_map_odom(2, 3), 0.0, 1.0e-12,
                     "TreeLoc++ map-to-odom z must remain zero");
        RequireClose(estimated_map_odom_yaw,
                     expected_map_odom_yaw,
                     1.0e-5,
                     "map-to-odom yaw mismatch");
        RequireClose(result.lidar_update_planar, 0.0, 1.0e-5,
                     "unexpected LiDAR planar update");
        RequireClose(result.lidar_update_yaw, 0.0, 1.0e-5,
                     "unexpected LiDAR yaw update");
        Require(result.candidates.size() == 3,
                "unexpected candidate diagnostic count");

        std::cout << "treelocpp hypothesis localization API smoke test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "treelocpp hypothesis localization API smoke test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
