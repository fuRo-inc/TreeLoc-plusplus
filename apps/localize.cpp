#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include "treelocpp/config.h"
#include "treelocpp/descriptors.h"
#include "treelocpp/geometry.h"
#include "treelocpp/matching.h"
#include "treelocpp/types.h"

namespace {

struct Args {
    std::filesystem::path config_path;
    int query_idx = -1;
    int prior_idx = -1;
    double search_radius_m = 30.0;
    int top_k = 10;
    double prior_gate_xy_m = 3.0;
    double prior_gate_yaw_rad = 0.3;
    bool exclude_self = false;
};

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " CONFIG "
        << "--query_idx N "
        << "[--prior_idx N] "
        << "[--search_radius M] "
        << "[--top_k K] "
        << "[--prior_gate_xy M] "
        << "[--prior_gate_yaw_rad RAD] "
        << "[--exclude_self]\n\n"
        << "Example:\n"
        << "  " << argv0 << " config/furo_relaxed_intra.yaml "
        << "--query_idx 541 --prior_idx 541 --search_radius 30 --top_k 20\n";
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
        } else if (key == "--search_radius") {
            args.search_radius_m = std::stod(need_value(key));
        } else if (key == "--top_k") {
            args.top_k = std::stoi(need_value(key));
        } else if (key == "--exclude_self") {
            args.exclude_self = true;
        } else if (key == "--prior_gate_xy") {
            args.prior_gate_xy_m = std::stod(need_value(key));
        } else if (key == "--prior_gate_yaw") {
            args.prior_gate_yaw_rad = std::stod(need_value(key));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.query_idx < 0) {
        throw std::runtime_error("--query_idx is required");
    }
    if (args.prior_idx < 0) {
        args.prior_idx = args.query_idx;
    }
    if (args.search_radius_m <= 0.0) {
        throw std::runtime_error("--search_radius must be positive");
    }
    if (args.top_k <= 0) {
        throw std::runtime_error("--top_k must be positive");
    }
    if (args.prior_gate_xy_m <= 0.0) {
        throw std::runtime_error("--prior_gate_xy must be positive");
    }
    if (args.prior_gate_yaw_rad <= 0.0) {
        throw std::runtime_error("--prior_gate_yaw must be positive");
    }
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

// Same convention as pose-edge export in eval.cpp:
// q_idx db_idx overlap x y z roll pitch yaw
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
                     double prior_gate_xy_m,
                     double prior_gate_yaw_rad) {
    if (!res.transform.ok) {
        return {false, "transform_not_ok"};
    }
    if (res.transform.overlap < 0.10) {
        return {false, "low_overlap"};
    }
    if (res.transform.pairs.size() < 3) {
        return {false, "few_pairs"};
    }
    if (std::abs(rel_t.z()) > 3.0) {
        return {false, "bad_rel_z"};
    }
    if (std::abs(rel_roll) > 0.3) {
        return {false, "bad_roll"};
    }
    if (std::abs(rel_pitch) > 0.3) {
        return {false, "bad_pitch"};
    }
    if (std::abs(rel_yaw) > 0.5) {
        return {false, "bad_yaw"};
    }
    if (!std::isfinite(prior_err_xy) || prior_err_xy > prior_gate_xy_m) {
        return {false, "bad_prior_xy"};
    }
    if (!std::isfinite(prior_err_yaw) || prior_err_yaw > prior_gate_yaw_rad) {
        return {false, "bad_prior_yaw"};
    }
    return {true, "ok"};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);

        treelocpp::Config config;
        std::string error;
        if (!treelocpp::LoadConfig(args.config_path, config, &error)) {
            throw std::runtime_error(error);
        }

        config.mode = "intra";

        std::cerr << "Loading dataset: " << config.dataset_root << "\n";
        const treelocpp::Dataset dataset =
            treelocpp::LoadDataset(config.dataset_root, config, config.neighbor_past_only);

        const size_t query_slot = SlotOf(dataset, args.query_idx, "query");
        const size_t prior_slot = SlotOf(dataset, args.prior_idx, "prior");

        const auto& qf = dataset.frames[query_slot];
        const auto& prior = dataset.frames[prior_slot].pose;

        std::vector<size_t> candidate_slots;
        for (size_t i = 0; i < dataset.frames.size(); ++i) {
            const auto& cf = dataset.frames[i];

            if (args.exclude_self && cf.index == qf.index) {
                continue;
            }

            const double d = PoseDistanceXYLocal(cf.pose, prior);
            if (d <= args.search_radius_m) {
                candidate_slots.push_back(i);
            }
        }

        if (candidate_slots.empty()) {
            throw std::runtime_error("no candidate anchors within search radius");
        }

        std::cerr << "query_idx: " << args.query_idx << "\n";
        std::cerr << "prior_idx: " << args.prior_idx << "\n";
        std::cerr << "search_radius_m: " << args.search_radius_m << "\n";
        std::cerr << "candidate_count: " << candidate_slots.size() << "\n";

        auto results = treelocpp::RankCandidates(
            dataset,
            dataset,
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

        const int n = std::min<int>(args.top_k, static_cast<int>(results.size()));
        for (int r = 0; r < n; ++r) {
            const auto& res = results[r];

            const size_t cand_slot = SlotOf(dataset, res.candidate_index, "candidate");
            const auto& cf = dataset.frames[cand_slot];

            const double prior_dist = PoseDistanceXYLocal(cf.pose, prior);
            const double gt_dist = PoseDistanceXYLocal(cf.pose, qf.pose);

            double rel_roll = 0.0;
            double rel_pitch = 0.0;
            double rel_yaw = 0.0;
            Eigen::Vector3d rel_t = Eigen::Vector3d::Zero();

            const double nan = std::numeric_limits<double>::quiet_NaN();

            double est_x = nan;
            double est_y = nan;
            double est_z = nan;
            double est_yaw = nan;
            double gt_x = qf.pose.x;
            double gt_y = qf.pose.y;
            double gt_z = qf.pose.z;
            double gt_yaw = treelocpp::YawFromPose(qf.pose);
            double err_xy = nan;
            double err_z = nan;
            double err_yaw = nan;
            double prior_err_xy = nan;
            double prior_err_yaw = nan;
            
            if (res.transform.ok) {
                const Eigen::Matrix4d T_query_candidate =
                    PredictedRelativeTransform(qf, cf, res);

                rel_t = T_query_candidate.block<3, 1>(0, 3);
                treelocpp::EulerZYX(
                    T_query_candidate.block<3, 3>(0, 0),
                    rel_roll,
                    rel_pitch,
                    rel_yaw
                );

                const Eigen::Matrix4d T_map_candidate =
                    treelocpp::PoseToTransform(cf.pose);

                // PredictedRelativeTransform returns T_query_candidate.
                // Therefore:
                //   T_map_candidate = T_map_query * T_query_candidate
                //   T_map_query     = T_map_candidate * inverse(T_query_candidate)
                const Eigen::Matrix4d T_map_query_est =
                    T_map_candidate * T_query_candidate.inverse();

                const Eigen::Vector3d est_p =
                    T_map_query_est.block<3, 1>(0, 3);

                est_x = est_p.x();
                est_y = est_p.y();
                est_z = est_p.z();
                est_yaw = YawFromTransform(T_map_query_est);

                const double dx = est_x - gt_x;
                const double dy = est_y - gt_y;
                err_xy = std::sqrt(dx * dx + dy * dy);
                err_z = std::abs(est_z - gt_z);
                err_yaw = std::abs(treelocpp::WrapAngle(est_yaw - gt_yaw));
                const double pdx = est_x - prior.x;
                const double pdy = est_y - prior.y;
                prior_err_xy = std::sqrt(pdx * pdx + pdy * pdy);

                const double prior_yaw = treelocpp::YawFromPose(prior);
                prior_err_yaw = std::abs(treelocpp::WrapAngle(est_yaw - prior_yaw));
            }

            const GateResult gate =
                CheckGate(
                    res,
                    rel_t,
                    rel_roll,
                    rel_pitch,
                    rel_yaw,
                    prior_err_xy,
                    prior_err_yaw,
                    args.prior_gate_xy_m,
                    args.prior_gate_yaw_rad
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
