#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "treelocpp/config.h"
#include "treelocpp/descriptors.h"
#include "treelocpp/geometry.h"
#include "treelocpp/matching.h"
#include "treelocpp/types.h"

namespace treelocpp {
namespace {

struct Args {
    std::filesystem::path config_path;
    std::filesystem::path query_root;
    std::filesystem::path database_root;
    double prior_x = 0.0;
    double prior_y = 0.0;
    double prior_yaw = 0.0;
    bool has_prior = false;
    double search_radius = 30.0;
    int top_k = 10;
    double min_overlap = 0.10;
    int min_pairs = 3;
    double max_rel_z = 3.0;
    double max_rel_roll = 0.3;
    double max_rel_pitch = 0.3;
    double max_rel_yaw = 0.7;
    double prior_gate_xy = 10.0;
    double prior_gate_yaw = 0.7;
};

[[noreturn]] void Usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " CONFIG.yaml "
        << "--query_root query/current --database_root data/furo_relaxed "
        << "--prior_x X --prior_y Y --prior_yaw YAW [options]\n\n"
        << "Options:\n"
        << "  --search_radius M      Candidate anchor radius around prior. Default: 30\n"
        << "  --top_k N              Number of ranked rows to print. Default: 10\n"
        << "  --min_overlap V        Acceptance gate. Default: 0.10\n"
        << "  --min_pairs N          Acceptance gate. Default: 3\n"
        << "  --prior_gate_xy M      Acceptance gate. Default: 10\n"
        << "  --prior_gate_yaw RAD   Acceptance gate. Default: 0.7\n"
        << "  --max_rel_z M          Relative transform gate. Default: 3\n"
        << "  --max_rel_roll RAD     Relative transform gate. Default: 0.3\n"
        << "  --max_rel_pitch RAD    Relative transform gate. Default: 0.3\n"
        << "  --max_rel_yaw RAD      Relative transform gate. Default: 0.7\n";
    throw std::runtime_error("invalid arguments");
}

Args ParseArgs(int argc, char** argv) {
    if (argc < 2) Usage(argv[0]);
    Args args;
    args.config_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (key == "--query_root") args.query_root = need(key);
        else if (key == "--database_root") args.database_root = need(key);
        else if (key == "--prior_x") { args.prior_x = std::stod(need(key)); args.has_prior = true; }
        else if (key == "--prior_y") { args.prior_y = std::stod(need(key)); args.has_prior = true; }
        else if (key == "--prior_yaw") { args.prior_yaw = std::stod(need(key)); args.has_prior = true; }
        else if (key == "--search_radius") args.search_radius = std::stod(need(key));
        else if (key == "--top_k") args.top_k = std::stoi(need(key));
        else if (key == "--min_overlap") args.min_overlap = std::stod(need(key));
        else if (key == "--min_pairs") args.min_pairs = std::stoi(need(key));
        else if (key == "--prior_gate_xy") args.prior_gate_xy = std::stod(need(key));
        else if (key == "--prior_gate_yaw") args.prior_gate_yaw = std::stod(need(key));
        else if (key == "--max_rel_z") args.max_rel_z = std::stod(need(key));
        else if (key == "--max_rel_roll") args.max_rel_roll = std::stod(need(key));
        else if (key == "--max_rel_pitch") args.max_rel_pitch = std::stod(need(key));
        else if (key == "--max_rel_yaw") args.max_rel_yaw = std::stod(need(key));
        else Usage(argv[0]);
    }
    if (args.query_root.empty()) throw std::runtime_error("--query_root is required");
    if (!args.has_prior) throw std::runtime_error("--prior_x/--prior_y/--prior_yaw are required");
    return args;
}

Eigen::Matrix4d PredictedRelativeTransform(const FrameData& qf,
                                           const FrameData& cf,
                                           const CandidateResult& best) {
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

std::vector<size_t> PriorCandidates(const Dataset& database, const Args& args) {
    std::vector<size_t> out;
    const double r2 = args.search_radius * args.search_radius;
    for (size_t i = 0; i < database.frames.size(); ++i) {
        const Pose& p = database.frames[i].pose;
        const double dx = p.x - args.prior_x;
        const double dy = p.y - args.prior_y;
        if (dx * dx + dy * dy <= r2) out.push_back(i);
    }
    return out;
}

std::string RejectReason(const CandidateResult& r,
                         double rel_z,
                         double rel_roll,
                         double rel_pitch,
                         double rel_yaw,
                         double prior_err_xy,
                         double prior_err_yaw,
                         const Args& args) {
    if (!r.transform.ok) return "transform_not_ok";
    if (r.transform.overlap < args.min_overlap) return "low_overlap";
    if (static_cast<int>(r.transform.pairs.size()) < args.min_pairs) return "few_pairs";
    if (std::abs(rel_z) > args.max_rel_z) return "bad_rel_z";
    if (std::abs(rel_roll) > args.max_rel_roll) return "bad_rel_roll";
    if (std::abs(rel_pitch) > args.max_rel_pitch) return "bad_rel_pitch";
    if (std::abs(rel_yaw) > args.max_rel_yaw) return "bad_rel_yaw";
    if (prior_err_xy > args.prior_gate_xy) return "bad_prior_xy";
    if (prior_err_yaw > args.prior_gate_yaw) return "bad_prior_yaw";
    return "ok";
}

}  // namespace
}  // namespace treelocpp

int main(int argc, char** argv) {
    try {
        using namespace treelocpp;
        const Args args = ParseArgs(argc, argv);

        Config config;
        std::string error;
        if (!LoadConfig(args.config_path, config, &error)) {
            throw std::runtime_error("failed to load config: " + error);
        }
        RefreshDerivedConfig(config);

        const std::filesystem::path database_root = args.database_root.empty()
            ? config.dataset_root
            : args.database_root;

        Dataset query = LoadDataset(args.query_root, config, false);
        Dataset database = LoadDataset(database_root, config, config.neighbor_past_only);
        if (query.frames.empty()) throw std::runtime_error("query dataset has no usable frames");
        if (database.frames.empty()) throw std::runtime_error("database dataset has no usable frames");

        const size_t query_slot = 0;
        const FrameData& qf = query.frames[query_slot];
        const std::vector<size_t> candidates = PriorCandidates(database, args);
        if (candidates.empty()) throw std::runtime_error("no database candidates inside search radius");

        std::vector<CandidateResult> results = RankCandidates(query, database, query_slot, candidates, config);
        std::sort(results.begin(), results.end(), [](const CandidateResult& a, const CandidateResult& b) {
            if (a.transform.ok != b.transform.ok) return a.transform.ok > b.transform.ok;
            if (std::abs(a.transform.overlap - b.transform.overlap) > 1e-12) return a.transform.overlap > b.transform.overlap;
            return a.retrieval_score > b.retrieval_score;
        });

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "query_root=" << args.query_root
                  << " database_root=" << database_root
                  << " query_frames=" << query.frames.size()
                  << " database_frames=" << database.frames.size()
                  << " candidate_count=" << candidates.size() << "\n";
        std::cout << "rank q_idx db_idx accepted reject_reason overlap matched_pairs retrieval_score hash_score "
                  << "rel_x rel_y rel_z rel_roll rel_pitch rel_yaw "
                  << "est_x est_y est_z est_yaw prior_err_xy prior_err_yaw\n";

        const int n = std::min(args.top_k, static_cast<int>(results.size()));
        for (int i = 0; i < n; ++i) {
            const CandidateResult& r = results[i];
            auto it = database.frame_to_slot.find(r.candidate_index);
            if (it == database.frame_to_slot.end()) continue;
            const FrameData& cf = database.frames[it->second];

            Eigen::Matrix4d T_qc = Eigen::Matrix4d::Identity();
            if (r.transform.ok) T_qc = PredictedRelativeTransform(qf, cf, r);
            double rel_roll = 0.0;
            double rel_pitch = 0.0;
            double rel_yaw = 0.0;
            EulerZYX(T_qc.block<3, 3>(0, 0), rel_roll, rel_pitch, rel_yaw);
            const Eigen::Vector3d rel_t = T_qc.block<3, 1>(0, 3);

            const Eigen::Matrix4d T_mc = PoseToTransform(cf.pose);
            const Eigen::Matrix4d T_mq = r.transform.ok ? T_mc * T_qc.inverse() : Eigen::Matrix4d::Identity();
            double est_roll = 0.0;
            double est_pitch = 0.0;
            double est_yaw = 0.0;
            EulerZYX(T_mq.block<3, 3>(0, 0), est_roll, est_pitch, est_yaw);
            const Eigen::Vector3d est_t = T_mq.block<3, 1>(0, 3);

            const double dx = est_t.x() - args.prior_x;
            const double dy = est_t.y() - args.prior_y;
            const double prior_err_xy = std::hypot(dx, dy);
            const double prior_err_yaw = std::abs(WrapAngle(est_yaw - args.prior_yaw));
            const std::string reason = RejectReason(
                r, rel_t.z(), rel_roll, rel_pitch, rel_yaw, prior_err_xy, prior_err_yaw, args);
            const int accepted = reason == "ok" ? 1 : 0;

            std::cout << i << ' ' << r.query_index << ' ' << r.candidate_index << ' '
                      << "accepted=" << accepted << ' '
                      << reason << ' '
                      << r.transform.overlap << ' '
                      << r.transform.pairs.size() << ' '
                      << r.retrieval_score << ' '
                      << r.hash_score << ' '
                      << rel_t.x() << ' ' << rel_t.y() << ' ' << rel_t.z() << ' '
                      << rel_roll << ' ' << rel_pitch << ' ' << rel_yaw << ' '
                      << est_t.x() << ' ' << est_t.y() << ' ' << est_t.z() << ' ' << est_yaw << ' '
                      << prior_err_xy << ' ' << prior_err_yaw << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "treelocpp_localize error: " << e.what() << "\n";
        return 1;
    }
}
