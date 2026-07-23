#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    std::filesystem::path query_root;
    std::filesystem::path database_root;
    double prior_x = 0.0;
    double prior_y = 0.0;
    double prior_z = 0.0;
    double prior_yaw = 0.0;
    double search_radius = 30.0;
    int top_k = 20;
    double match_distance = 2.0;
    int min_pairs = 2;
    double min_overlap = 0.10;
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
        << "  --prior_z Z          default 0\n"
        << "  --search_radius M   default 30\n"
        << "  --top_k N           default 20\n"
        << "  --match_distance M  prior-assisted NN distance. default 2.0\n"
        << "  --min_pairs N       default 2\n"
        << "  --min_overlap V     default 0.10\n"
        << "  --prior_gate_xy M   default 10\n"
        << "  --prior_gate_yaw R  default 0.7\n";
    throw std::runtime_error("invalid arguments");
}

Args ParseArgs(int argc, char** argv) {
    if (argc < 2) Usage(argv[0]);
    Args args;
    args.config_path = argv[1];
    bool px = false, py = false, pyaw = false;
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (key == "--query_root") args.query_root = need(key);
        else if (key == "--database_root") args.database_root = need(key);
        else if (key == "--prior_x") { args.prior_x = std::stod(need(key)); px = true; }
        else if (key == "--prior_y") { args.prior_y = std::stod(need(key)); py = true; }
        else if (key == "--prior_z") args.prior_z = std::stod(need(key));
        else if (key == "--prior_yaw") { args.prior_yaw = std::stod(need(key)); pyaw = true; }
        else if (key == "--search_radius") args.search_radius = std::stod(need(key));
        else if (key == "--top_k") args.top_k = std::stoi(need(key));
        else if (key == "--match_distance") args.match_distance = std::stod(need(key));
        else if (key == "--min_pairs") args.min_pairs = std::stoi(need(key));
        else if (key == "--min_overlap") args.min_overlap = std::stod(need(key));
        else if (key == "--prior_gate_xy") args.prior_gate_xy = std::stod(need(key));
        else if (key == "--prior_gate_yaw") args.prior_gate_yaw = std::stod(need(key));
        else Usage(argv[0]);
    }
    if (args.query_root.empty()) throw std::runtime_error("--query_root is required");
    if (args.database_root.empty()) throw std::runtime_error("--database_root is required");
    if (!px || !py || !pyaw) throw std::runtime_error("--prior_x, --prior_y, and --prior_yaw are required");
    return args;
}

Eigen::Matrix4d PoseFromXYYaw(double x, double y, double z, double yaw) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    T(0, 0) = c;
    T(0, 1) = -s;
    T(1, 0) = s;
    T(1, 1) = c;
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
}

std::vector<size_t> CandidateSlots(const treelocpp::Dataset& database, const Args& args) {
    std::vector<size_t> slots;
    const double r2 = args.search_radius * args.search_radius;
    for (size_t i = 0; i < database.frames.size(); ++i) {
        const auto& p = database.frames[i].pose;
        const double dx = p.x - args.prior_x;
        const double dy = p.y - args.prior_y;
        if (dx * dx + dy * dy <= r2) slots.push_back(i);
    }
    return slots;
}

double YawFromMatrix(const Eigen::Matrix3d& R) {
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    treelocpp::EulerZYX(R, roll, pitch, yaw);
    return yaw;
}

struct MatchPair {
    Eigen::Vector2d q;
    Eigen::Vector2d c;
    double dist = 0.0;
};

Eigen::Matrix3d EstimateRigid2D(const std::vector<MatchPair>& pairs) {
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    if (pairs.empty()) return T;
    Eigen::Vector2d q_mean = Eigen::Vector2d::Zero();
    Eigen::Vector2d c_mean = Eigen::Vector2d::Zero();
    for (const auto& p : pairs) {
        q_mean += p.q;
        c_mean += p.c;
    }
    q_mean /= static_cast<double>(pairs.size());
    c_mean /= static_cast<double>(pairs.size());

    if (pairs.size() == 1) {
        T.block<2, 1>(0, 2) = c_mean - q_mean;
        return T;
    }

    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    for (const auto& p : pairs) {
        H += (p.q - q_mean) * (p.c - c_mean).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0.0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1.0;
        R = V * svd.matrixU().transpose();
    }
    const Eigen::Vector2d t = c_mean - R * q_mean;
    T.block<2, 2>(0, 0) = R;
    T.block<2, 1>(0, 2) = t;
    return T;
}

struct FallbackResult {
    int pairs = 0;
    double overlap = 0.0;
    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
    double mean_residual = std::numeric_limits<double>::quiet_NaN();
    double max_residual = std::numeric_limits<double>::quiet_NaN();
    double prior_err_xy = std::numeric_limits<double>::quiet_NaN();
    double prior_err_yaw = std::numeric_limits<double>::quiet_NaN();
};

FallbackResult PriorAssistedNN(const treelocpp::FrameData& qf,
                               const treelocpp::FrameData& cf,
                               const Eigen::Matrix4d& T_map_prior,
                               double prior_yaw,
                               double match_distance) {
    FallbackResult out;
    const Eigen::Matrix4d T_map_candidate = treelocpp::PoseToTransform(cf.pose);
    const Eigen::Matrix4d T_candidate_query_seed = T_map_candidate.inverse() * T_map_prior;
    const Eigen::Matrix2d R_seed = T_candidate_query_seed.block<2, 2>(0, 0);
    const Eigen::Vector2d t_seed = T_candidate_query_seed.block<2, 1>(0, 3);

    std::vector<char> q_used(qf.trees.size(), 0);
    std::vector<MatchPair> pairs;
    pairs.reserve(std::min(qf.trees.size(), cf.trees.size()));

    for (const auto& ct : cf.trees) {
        const Eigen::Vector2d cp(ct.x, ct.y);
        int best = -1;
        double best_d2 = match_distance * match_distance;
        Eigen::Vector2d best_seed_q = Eigen::Vector2d::Zero();
        for (int qi = 0; qi < static_cast<int>(qf.trees.size()); ++qi) {
            if (q_used[qi]) continue;
            const auto& qt = qf.trees[qi];
            const Eigen::Vector2d qp_seed = R_seed * Eigen::Vector2d(qt.x, qt.y) + t_seed;
            const double d2 = (qp_seed - cp).squaredNorm();
            if (d2 <= best_d2) {
                best_d2 = d2;
                best = qi;
                best_seed_q = qp_seed;
            }
        }
        if (best >= 0) {
            q_used[best] = 1;
            const auto& qt = qf.trees[best];
            pairs.push_back({Eigen::Vector2d(qt.x, qt.y), cp, std::sqrt(best_d2)});
        }
    }

    out.pairs = static_cast<int>(pairs.size());
    const int uni = static_cast<int>(qf.trees.size() + cf.trees.size() - out.pairs);
    out.overlap = uni > 0 ? static_cast<double>(out.pairs) / static_cast<double>(uni) : 0.0;

    Eigen::Matrix3d T_candidate_query_2d = Eigen::Matrix3d::Identity();
    if (pairs.size() >= 2) {
        T_candidate_query_2d = EstimateRigid2D(pairs);
    } else {
        T_candidate_query_2d.block<2, 2>(0, 0) = R_seed;
        T_candidate_query_2d.block<2, 1>(0, 2) = t_seed;
    }

    Eigen::Matrix4d T_candidate_query = Eigen::Matrix4d::Identity();
    T_candidate_query.block<2, 2>(0, 0) = T_candidate_query_2d.block<2, 2>(0, 0);
    T_candidate_query.block<2, 1>(0, 3) = T_candidate_query_2d.block<2, 1>(0, 2);
    T_candidate_query(2, 3) = T_candidate_query_seed(2, 3);
    out.T_map_query = T_map_candidate * T_candidate_query;

    if (!pairs.empty()) {
        double sum = 0.0;
        double mx = 0.0;
        const Eigen::Matrix2d R = T_candidate_query.block<2, 2>(0, 0);
        const Eigen::Vector2d t = T_candidate_query.block<2, 1>(0, 3);
        for (const auto& p : pairs) {
            const double e = (R * p.q + t - p.c).norm();
            sum += e;
            mx = std::max(mx, e);
        }
        out.mean_residual = sum / static_cast<double>(pairs.size());
        out.max_residual = mx;
    }

    const Eigen::Vector3d est = out.T_map_query.block<3, 1>(0, 3);
    out.prior_err_xy = std::hypot(est.x() - T_map_prior(0, 3), est.y() - T_map_prior(1, 3));
    out.prior_err_yaw = std::abs(treelocpp::WrapAngle(YawFromMatrix(out.T_map_query.block<3, 3>(0, 0)) - prior_yaw));
    return out;
}

struct Row {
    int db_idx = -1;
    double prior_dist = 0.0;
    double retrieval_score = std::numeric_limits<double>::quiet_NaN();
    int hash_score = 0;
    bool rank_transform_ok = false;
    int fallback_pairs = 0;
    double fallback_overlap = 0.0;
    double nn_mean_residual = std::numeric_limits<double>::quiet_NaN();
    double nn_max_residual = std::numeric_limits<double>::quiet_NaN();
    int accepted = 0;
    double est_x = std::numeric_limits<double>::quiet_NaN();
    double est_y = std::numeric_limits<double>::quiet_NaN();
    double est_z = std::numeric_limits<double>::quiet_NaN();
    double est_yaw = std::numeric_limits<double>::quiet_NaN();
    double prior_err_xy = std::numeric_limits<double>::quiet_NaN();
    double prior_err_yaw = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        treelocpp::Config config;
        std::string error;
        if (!treelocpp::LoadConfig(args.config_path, config, &error)) {
            throw std::runtime_error(error);
        }
        treelocpp::RefreshDerivedConfig(config);

        auto query = treelocpp::LoadDataset(args.query_root, config, false);
        auto database = treelocpp::LoadDataset(args.database_root, config, config.neighbor_past_only);
        if (query.frames.empty()) throw std::runtime_error("query dataset has no usable frames");
        if (database.frames.empty()) throw std::runtime_error("database dataset has no usable frames");

        const auto& qf = query.frames.front();
        const auto candidates = CandidateSlots(database, args);
        if (candidates.empty()) throw std::runtime_error("no database candidates inside search radius");

        auto ranked = treelocpp::RankCandidates(query, database, 0, candidates, config);
        std::unordered_map<int, treelocpp::CandidateResult> by_idx;
        for (const auto& r : ranked) by_idx[r.candidate_index] = r;

        const Eigen::Matrix4d T_map_prior = PoseFromXYYaw(args.prior_x, args.prior_y, args.prior_z, args.prior_yaw);
        std::vector<Row> rows;
        rows.reserve(candidates.size());
        for (size_t slot : candidates) {
            const auto& cf = database.frames[slot];
            Row row;
            row.db_idx = cf.index;
            row.prior_dist = std::hypot(cf.pose.x - args.prior_x, cf.pose.y - args.prior_y);
            auto it = by_idx.find(cf.index);
            if (it != by_idx.end()) {
                row.retrieval_score = it->second.retrieval_score;
                row.hash_score = it->second.hash_score;
                row.rank_transform_ok = it->second.transform.ok;
            }
            const FallbackResult fb = PriorAssistedNN(qf, cf, T_map_prior, args.prior_yaw, args.match_distance);
            row.fallback_pairs = fb.pairs;
            row.fallback_overlap = fb.overlap;
            row.nn_mean_residual = fb.mean_residual;
            row.nn_max_residual = fb.max_residual;
            row.est_x = fb.T_map_query(0, 3);
            row.est_y = fb.T_map_query(1, 3);
            row.est_z = fb.T_map_query(2, 3);
            row.est_yaw = YawFromMatrix(fb.T_map_query.block<3, 3>(0, 0));
            row.prior_err_xy = fb.prior_err_xy;
            row.prior_err_yaw = fb.prior_err_yaw;
            row.accepted = (row.fallback_pairs >= args.min_pairs &&
                            row.fallback_overlap >= args.min_overlap &&
                            row.prior_err_xy <= args.prior_gate_xy &&
                            row.prior_err_yaw <= args.prior_gate_yaw) ? 1 : 0;
            rows.push_back(row);
        }

        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            if (a.accepted != b.accepted) return a.accepted > b.accepted;
            if (a.fallback_pairs != b.fallback_pairs) return a.fallback_pairs > b.fallback_pairs;
            if (std::abs(a.fallback_overlap - b.fallback_overlap) > 1e-12) return a.fallback_overlap > b.fallback_overlap;
            return a.prior_dist < b.prior_dist;
        });

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "query_root=" << args.query_root
                  << " database_root=" << args.database_root
                  << " query_trees=" << qf.trees.size()
                  << " database_frames=" << database.frames.size()
                  << " candidate_count=" << candidates.size()
                  << " match_distance=" << args.match_distance << "\n";
        std::cout << "rank db_idx accepted prior_dist retrieval_score hash_score rank_transform_ok "
                  << "nn_pairs nn_overlap nn_mean_residual nn_max_residual "
                  << "est_x est_y est_z est_yaw prior_err_xy prior_err_yaw\n";
        const int n = std::min(args.top_k, static_cast<int>(rows.size()));
        for (int i = 0; i < n; ++i) {
            const Row& r = rows[i];
            std::cout << i << ' '
                      << r.db_idx << ' '
                      << r.accepted << ' '
                      << r.prior_dist << ' '
                      << r.retrieval_score << ' '
                      << r.hash_score << ' '
                      << (r.rank_transform_ok ? 1 : 0) << ' '
                      << r.fallback_pairs << ' '
                      << r.fallback_overlap << ' '
                      << r.nn_mean_residual << ' '
                      << r.nn_max_residual << ' '
                      << r.est_x << ' '
                      << r.est_y << ' '
                      << r.est_z << ' '
                      << r.est_yaw << ' '
                      << r.prior_err_xy << ' '
                      << r.prior_err_yaw << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "treelocpp_localize_external error: " << e.what() << "\n";
        Usage(argv[0]);
        return 1;
    }
}
