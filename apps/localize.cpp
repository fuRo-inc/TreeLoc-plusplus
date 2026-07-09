#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "treelocpp/config.h"
#include "treelocpp/descriptors.h"
#include "treelocpp/geometry.h"
#include "treelocpp/matching.h"
#include "treelocpp/types.h"

namespace {

struct Args {
    std::filesystem::path config_path;

    // Intra-dataset mode. This preserves the previous localize.cpp workflow.
    int query_idx = -1;
    int prior_idx = -1;
    bool exclude_self = false;

    // External-query mode. Use query/current against a database DFI root.
    std::filesystem::path query_root;
    std::filesystem::path database_root;
    double prior_x = 0.0;
    double prior_y = 0.0;
    double prior_yaw = 0.0;
    bool has_prior_pose = false;

    double search_radius_m = 30.0;
    int top_k = 10;
    double prior_gate_xy_m = 3.0;
    double prior_gate_yaw_rad = 0.3;

    double min_overlap = 0.10;
    int min_pairs = 3;
    double max_rel_z = 3.0;
    double max_rel_roll = 0.3;
    double max_rel_pitch = 0.3;
    double max_rel_yaw = 0.5;
};

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage: intra dataset mode\n"
        << "  " << argv0 << " CONFIG "
        << "--query_idx N "
        << "[--prior_idx N] "
        << "[--search_radius M] "
        << "[--top_k K] "
        << "[--prior_gate_xy M] "
        << "[--prior_gate_yaw RAD] "
        << "[--exclude_self]\n\n"
        << "Example:\n"
        << "  " << argv0 << " config/furo_relaxed_intra.yaml "
        << "--query_idx 541 --prior_idx 541 --search_radius 30 --top_k 20\n\n"
        << "Usage: external query mode\n"
        << "  " << argv0 << " CONFIG "
        << "--query_root query/current "
        << "--database_root data/furo_relaxed "
        << "--prior_x X --prior_y Y --prior_yaw RAD "
        << "[--search_radius M] "
        << "[--top_k K] "
        << "[--prior_gate_xy M] "
        << "[--prior_gate_yaw RAD]\n\n"
        << "Example:\n"
        << "  " << argv0 << " config/furo_relaxed_intra.yaml "
        << "--query_root query/current --database_root data/furo_relaxed "
        << "--prior_x -670.165747430 --prior_y -26.559006413 --prior_yaw 1.878 "
        << "--search_radius 30 --top_k 10 --prior_gate_xy 10 --prior_gate_yaw 0.7\n";
}

Args ParseArgs(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        throw std::runtime_error("missing config path");
    }

    Args args;
    args.config_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];

        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--query_idx") {
            args.query_idx = std::stoi(need_value(key));
        } else if (key == "--prior_idx") {
            args.prior_idx = std::stoi(need_value(key));
        } else if (key == "--query_root") {
            args.query_root = need_value(key);
        } else if (key == "--database_root") {
            args.database_root = need_value(key);
        } else if (key == "--prior_x") {
            args.prior_x = std::stod(need_value(key));
            args.has_prior_pose = true;
        } else if (key == "--prior_y") {
            args.prior_y = std::stod(need_value(key));
            args.has_prior_pose = true;
        } else if (key == "--prior_yaw") {
            args.prior_yaw = std::stod(need_value(key));
            args.has_prior_pose = true;
        } else if (key == "--search_radius") {
            args.search_radius_m = std::stod(need_value(key));
        } else if (key == "--top_k") {
            args.top_k = std::stoi(need_value(key));
        } else if (key == "--exclude_self") {
            args.exclude_self = true;
        } else if (key == "--prior_gate_xy") {
            args.prior_gate_xy_m = std::stod(need_value(key));
        } else if (key == "--prior_gate_yaw" || key == "--prior_gate_yaw_rad") {
            args.prior_gate_yaw_rad = std::stod(need_value(key));
        } else if (key == "--min_overlap") {
            args.min_overlap = std::stod(need_value(key));
        } else if (key == "--min_pairs") {
            args.min_pairs = std::stoi(need_value(key));
        } else if (key == "--max_rel_z") {
            args.max_rel_z = std::stod(need_value(key));
        } else if (key == "--max_rel_roll") {
            args.max_rel_roll = std::stod(need_value(key));
        } else if (key == "--max_rel_pitch") {
            args.max_rel_pitch = std::stod(need_value(key));
        } else if (key == "--max_rel_yaw") {
            args.max_rel_yaw = std::stod(need_value(key));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    const bool external_mode = !args.query_root.empty();
    if (external_mode) {
        if (args.database_root.empty()) {
            throw std::runtime_error("--database_root is required when --query_root is used");
        }
        if (!args.has_prior_pose) {
            throw std::runtime_error("--prior_x, --prior_y, and --prior_yaw are required when --query_root is used");
        }
    } else {
        if (args.query_idx < 0) {
            throw std::runtime_error("--query_idx is required when --query_root is not used");
        }
        if (args.prior_idx < 0) {
            args.prior_idx = args.query_idx;
        }
    }

    if (args.search_radius_m <= 0.0) throw std::runtime_error("--search_radius must be positive");
    if (args.top_k <= 0) throw std::runtime_error("--top_k must be positive");
    if (args.prior_gate_xy_m <= 0.0) throw std::runtime_error("--prior_gate_xy must be positive");
    if (args.prior_gate_yaw_rad <= 0.0) throw std::runtime_error("--prior_gate_yaw must be positive");
    return args;
}

size_t SlotOf(const treelocpp::Dataset& dataset, int frame_idx, const std::string& label) {
    const auto it = dataset.frame_to_slot.find(frame_idx);
    if (it == dataset.frame_to_slot.end()) {
        throw std::runtime_error(label + " frame index not found: " + std::to_string(frame_idx));
    }
    return it->second;
}

double PoseDistanceXYLocal(const treelocpp::Pose& a, const treelocpp::Pose& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

Eigen::Matrix4d PredictedRelativeTransform(const treelocpp::FrameData& qf,
                                           const treelocpp::FrameData& cf,
                                           const treelocpp::CandidateResult& best) {
    Eigen::Matrix3d Rz_q2c = Eigen::Matrix3d::Identity();
    Rz_q2c.block<2, 2>(0, 0) = best.transform.R;

    const Eigen::Matrix3d R_rp_q2c =
        Eigen::AngleAxisd(best.vertical.axis_pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(best.vertical.axis_roll, Eigen::Vector3d::UnitX()).toRotationMatrix();

    const Eigen::Matrix3d R_q2c = Rz_q2c * R_rp_q2c;
    const Eigen::Vector3d t_q2c(best.transform.t.x(), best.transform.t.y(), 0.0);

    Eigen::Matrix3d R_c2q = R_q2c.transpose();
    Eigen::Vector3d t_c2q = -R_c2q * t_q2c;

    const Eigen::Matrix3d delta_r_c2q =
        Eigen::AngleAxisd(best.vertical.pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(best.vertical.roll, Eigen::Vector3d::UnitX()).toRotationMatrix();

    R_c2q = delta_r_c2q * R_c2q;
    t_c2q += Eigen::Vector3d(0.0, 0.0, best.vertical.z);

    Eigen::Matrix4d T_pred_aligned = Eigen::Matrix4d::Identity();
    T_pred_aligned.block<3, 3>(0, 0) = R_c2q;
    T_pred_aligned.block<3, 1>(0, 3) = t_c2q;

    return qf.alignment_transform.inverse() * T_pred_aligned * cf.alignment_transform;
}

double YawFromTransform(const Eigen::Matrix4d& T) {
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    treelocpp::EulerZYX(T.block<3, 3>(0, 0), roll, pitch, yaw);
    return yaw;
}

struct GateResult {
    bool accepted = false;
    std::string reason = "unknown";
};

GateResult CheckGate(const treelocpp::CandidateResult& res,
                     const Eigen::Vector3d& rel_t,
                     double rel_roll,
                     double rel_pitch,
                     double rel_yaw,
                     double prior_err_xy,
                     double prior_err_yaw,
                     const Args& args) {
    if (!res.transform.ok) return {false, "transform_not_ok"};
    if (res.transform.overlap < args.min_overlap) return {false, "low_overlap"};
    if (static_cast<int>(res.transform.pairs.size()) < args.min_pairs) return {false, "few_pairs"};
    if (std::abs(rel_t.z()) > args.max_rel_z) return {false, "bad_rel_z"};
    if (std::abs(rel_roll) > args.max_rel_roll) return {false, "bad_roll"};
    if (std::abs(rel_pitch) > args.max_rel_pitch) return {false, "bad_pitch"};
    if (std::abs(rel_yaw) > args.max_rel_yaw) return {false, "bad_yaw"};
    if (!std::isfinite(prior_err_xy) || prior_err_xy > args.prior_gate_xy_m) return {false, "bad_prior_xy"};
    if (!std::isfinite(prior_err_yaw) || prior_err_yaw > args.prior_gate_yaw_rad) return {false, "bad_prior_yaw"};
    return {true, "ok"};
}

std::vector<size_t> CandidateSlotsIntra(const treelocpp::Dataset& dataset,
                                        size_t query_slot,
                                        const treelocpp::Pose& prior,
                                        const Args& args) {
    std::vector<size_t> candidate_slots;
    const auto& qf = dataset.frames[query_slot];
    for (size_t i = 0; i < dataset.frames.size(); ++i) {
        const auto& cf = dataset.frames[i];
        if (args.exclude_self && cf.index == qf.index) continue;
        if (PoseDistanceXYLocal(cf.pose, prior) <= args.search_radius_m) candidate_slots.push_back(i);
    }
    return candidate_slots;
}

std::vector<size_t> CandidateSlotsExternal(const treelocpp::Dataset& database,
                                           const Args& args) {
    std::vector<size_t> candidate_slots;
    const double r2 = args.search_radius_m * args.search_radius_m;
    for (size_t i = 0; i < database.frames.size(); ++i) {
        const auto& p = database.frames[i].pose;
        const double dx = p.x - args.prior_x;
        const double dy = p.y - args.prior_y;
        if (dx * dx + dy * dy <= r2) candidate_slots.push_back(i);
    }
    return candidate_slots;
}

void PrintHeader() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout
        << "rank"
        << " q_idx"
        << " db_idx"
        << " accepted"
        << " reject_reason"
        << " overlap"
        << " matched_pairs"
        << " retrieval_score"
        << " hash_score"
        << " prior_dist"
        << " gt_dist"
        << " rel_x"
        << " rel_y"
        << " rel_z"
        << " rel_roll"
        << " rel_pitch"
        << " rel_yaw"
        << " est_x"
        << " est_y"
        << " est_z"
        << " est_yaw"
        << " gt_x"
        << " gt_y"
        << " gt_z"
        << " gt_yaw"
        << " err_xy"
        << " err_z"
        << " err_yaw"
        << " prior_err_xy"
        << " prior_err_yaw"
        << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const bool external_mode = !args.query_root.empty();

        treelocpp::Config config;
        std::string error;
        if (!treelocpp::LoadConfig(args.config_path, config, &error)) {
            throw std::runtime_error(error);
        }
        treelocpp::RefreshDerivedConfig(config);

        treelocpp::Dataset query_dataset;
        treelocpp::Dataset database_dataset;
        size_t query_slot = 0;
        std::vector<size_t> candidate_slots;
        treelocpp::Pose prior_pose;
        bool has_gt_pose = false;

        if (external_mode) {
            std::cerr << "Loading query dataset: " << args.query_root << "\n";
            query_dataset = treelocpp::LoadDataset(args.query_root, config, false);
            std::cerr << "Loading database dataset: " << args.database_root << "\n";
            database_dataset = treelocpp::LoadDataset(args.database_root, config, config.neighbor_past_only);
            if (query_dataset.frames.empty()) throw std::runtime_error("query dataset has no usable frames");
            if (database_dataset.frames.empty()) throw std::runtime_error("database dataset has no usable frames");
            query_slot = 0;
            prior_pose.x = args.prior_x;
            prior_pose.y = args.prior_y;
            prior_pose.z = 0.0;
            const double half = 0.5 * args.prior_yaw;
            prior_pose.qz = std::sin(half);
            prior_pose.qw = std::cos(half);
            candidate_slots = CandidateSlotsExternal(database_dataset, args);
            std::cerr << "mode: external\n";
            std::cerr << "query_root: " << args.query_root << "\n";
            std::cerr << "database_root: " << args.database_root << "\n";
        } else {
            config.mode = "intra";
            std::cerr << "Loading dataset: " << config.dataset_root << "\n";
            query_dataset = treelocpp::LoadDataset(config.dataset_root, config, config.neighbor_past_only);
            database_dataset = query_dataset;
            query_slot = SlotOf(query_dataset, args.query_idx, "query");
            const size_t prior_slot = SlotOf(query_dataset, args.prior_idx, "prior");
            prior_pose = query_dataset.frames[prior_slot].pose;
            candidate_slots = CandidateSlotsIntra(database_dataset, query_slot, prior_pose, args);
            has_gt_pose = true;
            std::cerr << "mode: intra\n";
            std::cerr << "query_idx: " << args.query_idx << "\n";
            std::cerr << "prior_idx: " << args.prior_idx << "\n";
        }

        if (candidate_slots.empty()) throw std::runtime_error("no candidate anchors within search radius");

        const auto& qf = query_dataset.frames[query_slot];
        std::cerr << "search_radius_m: " << args.search_radius_m << "\n";
        std::cerr << "candidate_count: " << candidate_slots.size() << "\n";

        auto results = treelocpp::RankCandidates(
            query_dataset,
            database_dataset,
            query_slot,
            candidate_slots,
            config
        );

        std::sort(
            results.begin(),
            results.end(),
            [](const treelocpp::CandidateResult& a, const treelocpp::CandidateResult& b) {
                if (a.transform.ok != b.transform.ok) return a.transform.ok > b.transform.ok;
                if (a.transform.overlap != b.transform.overlap) return a.transform.overlap > b.transform.overlap;
                return a.retrieval_score > b.retrieval_score;
            }
        );

        PrintHeader();

        const int n = std::min<int>(args.top_k, static_cast<int>(results.size()));
        for (int r = 0; r < n; ++r) {
            const auto& res = results[r];
            const size_t cand_slot = SlotOf(database_dataset, res.candidate_index, "candidate");
            const auto& cf = database_dataset.frames[cand_slot];

            const double prior_dist = PoseDistanceXYLocal(cf.pose, prior_pose);
            const double gt_dist = has_gt_pose ? PoseDistanceXYLocal(cf.pose, qf.pose) : std::numeric_limits<double>::quiet_NaN();

            double rel_roll = 0.0;
            double rel_pitch = 0.0;
            double rel_yaw = 0.0;
            Eigen::Vector3d rel_t = Eigen::Vector3d::Zero();

            const double nan = std::numeric_limits<double>::quiet_NaN();
            double est_x = nan;
            double est_y = nan;
            double est_z = nan;
            double est_yaw = nan;
            double gt_x = has_gt_pose ? qf.pose.x : nan;
            double gt_y = has_gt_pose ? qf.pose.y : nan;
            double gt_z = has_gt_pose ? qf.pose.z : nan;
            double gt_yaw = has_gt_pose ? treelocpp::YawFromPose(qf.pose) : nan;
            double err_xy = nan;
            double err_z = nan;
            double err_yaw = nan;
            double prior_err_xy = nan;
            double prior_err_yaw = nan;

            if (res.transform.ok) {
                const Eigen::Matrix4d T_query_candidate = PredictedRelativeTransform(qf, cf, res);
                rel_t = T_query_candidate.block<3, 1>(0, 3);
                treelocpp::EulerZYX(T_query_candidate.block<3, 3>(0, 0), rel_roll, rel_pitch, rel_yaw);

                const Eigen::Matrix4d T_map_candidate = treelocpp::PoseToTransform(cf.pose);
                const Eigen::Matrix4d T_map_query_est = T_map_candidate * T_query_candidate.inverse();
                const Eigen::Vector3d est_p = T_map_query_est.block<3, 1>(0, 3);
                est_x = est_p.x();
                est_y = est_p.y();
                est_z = est_p.z();
                est_yaw = YawFromTransform(T_map_query_est);

                if (has_gt_pose) {
                    const double dx = est_x - gt_x;
                    const double dy = est_y - gt_y;
                    err_xy = std::sqrt(dx * dx + dy * dy);
                    err_z = std::abs(est_z - gt_z);
                    err_yaw = std::abs(treelocpp::WrapAngle(est_yaw - gt_yaw));
                }
                const double pdx = est_x - prior_pose.x;
                const double pdy = est_y - prior_pose.y;
                prior_err_xy = std::sqrt(pdx * pdx + pdy * pdy);
                prior_err_yaw = std::abs(treelocpp::WrapAngle(est_yaw - treelocpp::YawFromPose(prior_pose)));
            }

            const GateResult gate = CheckGate(
                res,
                rel_t,
                rel_roll,
                rel_pitch,
                rel_yaw,
                prior_err_xy,
                prior_err_yaw,
                args
            );

            std::cout
                << r
                << " " << res.query_index
                << " " << res.candidate_index
                << " " << (gate.accepted ? 1 : 0)
                << " " << gate.reason
                << " " << res.transform.overlap
                << " " << res.transform.pairs.size()
                << " " << res.retrieval_score
                << " " << res.hash_score
                << " " << prior_dist
                << " " << gt_dist
                << " " << rel_t.x()
                << " " << rel_t.y()
                << " " << rel_t.z()
                << " " << rel_roll
                << " " << rel_pitch
                << " " << rel_yaw
                << " " << est_x
                << " " << est_y
                << " " << est_z
                << " " << est_yaw
                << " " << gt_x
                << " " << gt_y
                << " " << gt_z
                << " " << gt_yaw
                << " " << err_xy
                << " " << err_z
                << " " << err_yaw
                << " " << prior_err_xy
                << " " << prior_err_yaw
                << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "treelocpp_localize error: " << e.what() << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
}
