#pragma once

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "treelocpp/config.h"
#include "treelocpp/types.h"

namespace treelocpp {

struct HypothesisLocalizationOptions {
    double search_radius = 30.0;
    double match_distance = 1.5;
    int refine_iterations = 5;
    double dbh_soft_weight = 0.0;

    double triangle_edge_tolerance = 0.5;
    int triangle_max_hypotheses = 500;

    double consensus_xy = 2.0;
    double consensus_yaw = 0.15;
    int min_consensus_support = 3;
    int min_consensus_margin = 1;

    int min_pairs = 6;
    double min_query_coverage = 0.40;
    double min_overlap = 0.0;
    double max_mean_residual = 0.35;
    double max_max_residual = 0.85;

    double prior_gate_xy = 10.0;
    double prior_gate_yaw = 0.7;
};

struct HypothesisCandidateDiagnostic {
    int database_index = -1;
    double prior_distance = std::numeric_limits<double>::quiet_NaN();
    double retrieval_score = std::numeric_limits<double>::quiet_NaN();
    int hash_score = 0;
    bool rank_transform_ok = false;

    int triangle_hypothesis_count = 0;
    int triangle_consensus_support = 0;

    bool intrinsic_ok = false;
    bool consensus_member = false;
    bool final_selected = false;
    bool accepted = false;

    std::string reject_reason = "unknown";
    std::string selected_source;

    int prior_pairs = 0;
    double prior_mean_residual =
        std::numeric_limits<double>::quiet_NaN();

    int treeloc_pairs = 0;
    double treeloc_mean_residual =
        std::numeric_limits<double>::quiet_NaN();

    int triangle_pairs = 0;
    double triangle_mean_residual =
        std::numeric_limits<double>::quiet_NaN();

    int pairs = 0;
    double overlap = 0.0;
    double query_coverage = 0.0;
    double candidate_coverage = 0.0;
    double mean_residual = std::numeric_limits<double>::quiet_NaN();
    double max_residual = std::numeric_limits<double>::quiet_NaN();
    double mean_dbh_difference =
        std::numeric_limits<double>::quiet_NaN();

    double prior_error_xy = std::numeric_limits<double>::quiet_NaN();
    double prior_error_yaw = std::numeric_limits<double>::quiet_NaN();

    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
};

struct HypothesisLocalizationResult {
    bool ok = false;
    bool ambiguous = false;

    int query_index = -1;
    int map_index = -1;
    int candidate_count = 0;
    int support = 0;
    int runner_up_support = 0;

    std::string status = "uninitialized";

    std::vector<int> consensus_database_indices;

    int pairs = 0;
    double overlap = std::numeric_limits<double>::quiet_NaN();
    double query_coverage = std::numeric_limits<double>::quiet_NaN();
    double candidate_coverage = std::numeric_limits<double>::quiet_NaN();
    double mean_residual = std::numeric_limits<double>::quiet_NaN();
    double max_residual = std::numeric_limits<double>::quiet_NaN();

    // TreeLoc++ estimates only map-frame x/y/yaw. The z component of
    // T_map_query is copied from map_query_prior. Roll and pitch are zero.
    // These transforms must only be consumed when ok is true.
    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_map_odom = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_map_odom_prior = Eigen::Matrix4d::Identity();

    // Motion of the current LiDAR pose caused by applying the candidate
    // map-to-odom transform. Use these values for bounded-update safety.
    double lidar_update_planar =
        std::numeric_limits<double>::quiet_NaN();
    double lidar_update_yaw =
        std::numeric_limits<double>::quiet_NaN();

    // Direct map-to-odom parameter differences are diagnostic only.
    double parameter_planar =
        std::numeric_limits<double>::quiet_NaN();
    double parameter_yaw =
        std::numeric_limits<double>::quiet_NaN();

    std::vector<HypothesisCandidateDiagnostic> candidates;
};

HypothesisLocalizationResult LocalizeHypotheses(
    const FrameData& query,
    const Dataset& database,
    const Pose& map_query_prior,
    const Config& config,
    const HypothesisLocalizationOptions& options = {});

// Compatibility entry point used by the existing command-line application.
int RunLocalizeHypothesesCli(int argc, char** argv);

}  // namespace treelocpp
