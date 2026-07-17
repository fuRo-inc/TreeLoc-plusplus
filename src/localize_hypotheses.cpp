#include <algorithm>
#include <array>
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
#include "treelocpp/localize_hypotheses.h"

namespace {

constexpr std::array<std::array<int, 3>, 6> kTrianglePermutations{{
    {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
    {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}
}};

struct Args : treelocpp::HypothesisLocalizationOptions {
    std::filesystem::path config_path;
    std::filesystem::path query_root;
    std::filesystem::path database_root;

    double prior_x = 0.0;
    double prior_y = 0.0;
    double prior_z = 0.0;
    double prior_yaw = 0.0;

    // This only limits CLI diagnostic rows. It is not a localization gate.
    int top_k = 20;
};

[[noreturn]] void Usage(const char* argv0) {
    std::cerr
        << "Usage:\n  " << argv0 << " CONFIG.yaml "
        << "--query_root ROOT --database_root ROOT "
        << "--prior_x X --prior_y Y --prior_yaw RAD [options]\n\n"
        << "Options:\n"
        << "  --prior_z Z\n"
        << "  --search_radius M\n"
        << "  --top_k N\n"
        << "  --match_distance M\n"
        << "  --refine_iterations N\n"
        << "  --dbh_soft_weight W\n"
        << "  --triangle_edge_tolerance M\n"
        << "  --triangle_max_hypotheses N\n"
        << "  --consensus_xy M\n"
        << "  --consensus_yaw RAD\n"
        << "  --min_consensus_support N\n"
        << "  --min_consensus_margin N\n"
        << "  --min_pairs N\n"
        << "  --min_query_coverage V\n"
        << "  --min_overlap V\n"
        << "  --max_mean_residual M\n"
        << "  --max_max_residual M\n"
        << "  --prior_gate_xy M\n"
        << "  --prior_gate_yaw RAD\n";
    throw std::runtime_error("invalid arguments");
}

Args ParseArgs(int argc, char** argv) {
    if (argc < 2) Usage(argv[0]);
    Args args;
    args.config_path = argv[1];
    bool px = false;
    bool py = false;
    bool pyaw = false;
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
        else if (key == "--refine_iterations") args.refine_iterations = std::stoi(need(key));
        else if (key == "--dbh_soft_weight") args.dbh_soft_weight = std::stod(need(key));
        else if (key == "--triangle_edge_tolerance") {
            args.triangle_edge_tolerance = std::stod(need(key));
        } else if (key == "--triangle_max_hypotheses") {
            args.triangle_max_hypotheses = std::stoi(need(key));
        } else if (key == "--consensus_xy") args.consensus_xy = std::stod(need(key));
        else if (key == "--consensus_yaw") args.consensus_yaw = std::stod(need(key));
        else if (key == "--min_consensus_support") {
            args.min_consensus_support = std::stoi(need(key));
        }
        else if (key == "--min_consensus_margin") {
            args.min_consensus_margin = std::stoi(need(key));
        }
        else if (key == "--min_pairs") args.min_pairs = std::stoi(need(key));
        else if (key == "--min_query_coverage") {
            args.min_query_coverage = std::stod(need(key));
        }
        else if (key == "--min_overlap") args.min_overlap = std::stod(need(key));
        else if (key == "--max_mean_residual") args.max_mean_residual = std::stod(need(key));
        else if (key == "--max_max_residual") args.max_max_residual = std::stod(need(key));
        else if (key == "--prior_gate_xy") args.prior_gate_xy = std::stod(need(key));
        else if (key == "--prior_gate_yaw") args.prior_gate_yaw = std::stod(need(key));
        else Usage(argv[0]);
    }
    if (args.query_root.empty()) throw std::runtime_error("--query_root is required");
    if (args.database_root.empty()) throw std::runtime_error("--database_root is required");
    if (!px || !py || !pyaw) {
        throw std::runtime_error("--prior_x, --prior_y, and --prior_yaw are required");
    }
    if (args.match_distance <= 0.0) throw std::runtime_error("--match_distance must be positive");
    if (args.refine_iterations < 0) throw std::runtime_error("--refine_iterations must be non-negative");
    if (args.triangle_edge_tolerance <= 0.0) {
        throw std::runtime_error("--triangle_edge_tolerance must be positive");
    }
    if (args.triangle_max_hypotheses <= 0) {
        throw std::runtime_error("--triangle_max_hypotheses must be positive");
    }
    if (args.consensus_xy <= 0.0 || args.consensus_yaw <= 0.0) {
        throw std::runtime_error("consensus thresholds must be positive");
    }
    if (args.min_consensus_support <= 0) {
        throw std::runtime_error("--min_consensus_support must be positive");
    }
    if (args.min_consensus_margin < 0) {
        throw std::runtime_error("--min_consensus_margin must be non-negative");
    }
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

double YawFromMatrix(const Eigen::Matrix3d& R) {
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    treelocpp::EulerZYX(R, roll, pitch, yaw);
    return yaw;
}

struct PoseSummary {
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
};

PoseSummary SummarizePose(const Eigen::Matrix4d& transform) {
    PoseSummary summary;
    summary.translation = transform.block<3, 1>(0, 3);
    treelocpp::EulerZYX(
        transform.block<3, 3>(0, 0),
        summary.roll,
        summary.pitch,
        summary.yaw
    );
    return summary;
}

double TreeRadius(const treelocpp::Tree& tree) {
    return tree.dbh_valid && std::isfinite(tree.dbh)
        ? tree.dbh
        : std::numeric_limits<double>::quiet_NaN();
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

struct Pair {
    int query = -1;
    int candidate = -1;
    double distance = 0.0;
    double dbh_difference = 0.0;
    double cost = 0.0;
};

std::vector<Pair> AssociateOneToOne(const treelocpp::FrameData& query,
                                    const treelocpp::FrameData& candidate,
                                    const Eigen::Matrix2d& R,
                                    const Eigen::Vector2d& t,
                                    double max_distance,
                                    double dbh_soft_weight) {
    std::vector<Pair> edges;
    for (int qi = 0; qi < static_cast<int>(query.trees.size()); ++qi) {
        const Eigen::Vector2d qp =
            R * Eigen::Vector2d(query.trees[qi].x, query.trees[qi].y) + t;
        for (int ci = 0; ci < static_cast<int>(candidate.trees.size()); ++ci) {
            const Eigen::Vector2d cp(candidate.trees[ci].x, candidate.trees[ci].y);
            const double distance = (qp - cp).norm();
            if (distance > max_distance) continue;
            const double q_dbh = TreeRadius(query.trees[qi]);
            const double c_dbh = TreeRadius(candidate.trees[ci]);
            double dbh_difference = 0.0;
            if (std::isfinite(q_dbh) && std::isfinite(c_dbh)) {
                dbh_difference = std::abs(q_dbh - c_dbh);
            }
            const double clipped_dbh = std::min(dbh_difference, 1.0);
            const double cost = distance + dbh_soft_weight * clipped_dbh;
            edges.push_back({qi, ci, distance, dbh_difference, cost});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const Pair& a, const Pair& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.distance < b.distance;
    });
    std::vector<char> query_used(query.trees.size(), 0);
    std::vector<char> candidate_used(candidate.trees.size(), 0);
    std::vector<Pair> pairs;
    for (const auto& edge : edges) {
        if (query_used[edge.query] || candidate_used[edge.candidate]) continue;
        query_used[edge.query] = 1;
        candidate_used[edge.candidate] = 1;
        pairs.push_back(edge);
    }
    return pairs;
}

bool EstimateRigid2D(const treelocpp::FrameData& query,
                     const treelocpp::FrameData& candidate,
                     const std::vector<Pair>& pairs,
                     Eigen::Matrix2d& R,
                     Eigen::Vector2d& t) {
    if (pairs.size() < 2) return false;
    Eigen::MatrixXd Q(pairs.size(), 2);
    Eigen::MatrixXd C(pairs.size(), 2);
    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        Q.row(i) << query.trees[pairs[i].query].x, query.trees[pairs[i].query].y;
        C.row(i) << candidate.trees[pairs[i].candidate].x,
                    candidate.trees[pairs[i].candidate].y;
    }
    const Eigen::Vector2d q_mean = Q.colwise().mean();
    const Eigen::Vector2d c_mean = C.colwise().mean();
    Q.rowwise() -= q_mean.transpose();
    C.rowwise() -= c_mean.transpose();
    const Eigen::Matrix2d H = Q.transpose() * C;
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d V = svd.matrixV();
    R = V * svd.matrixU().transpose();
    if (R.determinant() < 0.0) {
        V.col(1) *= -1.0;
        R = V * svd.matrixU().transpose();
    }
    t = c_mean - R * q_mean;
    return true;
}

std::array<double, 3> TriangleSides(
    const treelocpp::FrameData& frame,
    const std::array<int, 3>& triangle
) {
    std::array<double, 3> sides{{
        (frame.centers[triangle[0]] - frame.centers[triangle[1]]).norm(),
        (frame.centers[triangle[1]] - frame.centers[triangle[2]]).norm(),
        (frame.centers[triangle[2]] - frame.centers[triangle[0]]).norm()
    }};
    std::sort(sides.begin(), sides.end());
    return sides;
}

struct TriangleSeed {
    Eigen::Matrix4d T_candidate_query = Eigen::Matrix4d::Identity();
    double score = std::numeric_limits<double>::infinity();
    double edge_difference = std::numeric_limits<double>::infinity();
    double triangle_residual = std::numeric_limits<double>::infinity();
    double mean_dbh_difference = std::numeric_limits<double>::infinity();
};

std::vector<TriangleSeed> BuildTolerantTriangleSeeds(
    const treelocpp::FrameData& query,
    const treelocpp::FrameData& candidate,
    const Args& args
) {
    std::vector<TriangleSeed> seeds;
    for (const auto& query_triangle : query.triangles.simplices) {
        const auto query_sides = TriangleSides(query, query_triangle);
        for (const auto& candidate_triangle : candidate.triangles.simplices) {
            const auto candidate_sides = TriangleSides(candidate, candidate_triangle);
            double edge_difference = 0.0;
            bool edge_ok = true;
            for (int i = 0; i < 3; ++i) {
                const double difference = std::abs(query_sides[i] - candidate_sides[i]);
                edge_difference += difference;
                if (difference > args.triangle_edge_tolerance) edge_ok = false;
            }
            if (!edge_ok) continue;

            TriangleSeed best;
            for (const auto& permutation : kTrianglePermutations) {
                std::vector<Pair> pairs;
                pairs.reserve(3);
                double dbh_sum = 0.0;
                for (int i = 0; i < 3; ++i) {
                    const int qi = query_triangle[i];
                    const int ci = candidate_triangle[permutation[i]];
                    const double q_dbh = TreeRadius(query.trees[qi]);
                    const double c_dbh = TreeRadius(candidate.trees[ci]);
                    const double dbh_difference =
                        std::isfinite(q_dbh) && std::isfinite(c_dbh)
                        ? std::abs(q_dbh - c_dbh)
                        : 0.0;
                    dbh_sum += dbh_difference;
                    pairs.push_back({qi, ci, 0.0, dbh_difference, 0.0});
                }
                Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
                Eigen::Vector2d t = Eigen::Vector2d::Zero();
                if (!EstimateRigid2D(query, candidate, pairs, R, t)) continue;
                double residual_sum = 0.0;
                for (const auto& pair : pairs) {
                    const Eigen::Vector2d qp =
                        R * Eigen::Vector2d(
                            query.trees[pair.query].x,
                            query.trees[pair.query].y
                        ) + t;
                    const Eigen::Vector2d cp(
                        candidate.trees[pair.candidate].x,
                        candidate.trees[pair.candidate].y
                    );
                    residual_sum += (qp - cp).norm();
                }
                const double triangle_residual = residual_sum / 3.0;
                const double mean_dbh_difference = dbh_sum / 3.0;
                const double score =
                    edge_difference + triangle_residual
                    + args.dbh_soft_weight * std::min(mean_dbh_difference, 1.0);
                if (score >= best.score) continue;
                best.score = score;
                best.edge_difference = edge_difference;
                best.triangle_residual = triangle_residual;
                best.mean_dbh_difference = mean_dbh_difference;
                best.T_candidate_query.block<2, 2>(0, 0) = R;
                best.T_candidate_query.block<2, 1>(0, 3) = t;
            }
            if (std::isfinite(best.score)) seeds.push_back(best);
        }
    }
    std::sort(seeds.begin(), seeds.end(), [](const TriangleSeed& a, const TriangleSeed& b) {
        return a.score < b.score;
    });
    if (static_cast<int>(seeds.size()) > args.triangle_max_hypotheses) {
        seeds.resize(args.triangle_max_hypotheses);
    }
    return seeds;
}

struct HypothesisResult {
    std::string source;
    bool valid = false;
    int pairs = 0;
    double overlap = 0.0;
    double query_coverage = 0.0;
    double candidate_coverage = 0.0;
    double mean_residual = std::numeric_limits<double>::quiet_NaN();
    double max_residual = std::numeric_limits<double>::quiet_NaN();
    double mean_dbh_difference = std::numeric_limits<double>::quiet_NaN();
    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
    double prior_err_xy = std::numeric_limits<double>::quiet_NaN();
    double prior_err_yaw = std::numeric_limits<double>::quiet_NaN();
};

HypothesisResult RefineHypothesis(const treelocpp::FrameData& query,
                                  const treelocpp::FrameData& candidate,
                                  const Eigen::Matrix4d& T_candidate_query_seed,
                                  const Eigen::Matrix4d& T_map_prior,
                                  double prior_yaw,
                                  const std::string& source,
                                  const Args& args) {
    HypothesisResult out;
    out.source = source;
    Eigen::Matrix2d R = T_candidate_query_seed.block<2, 2>(0, 0);
    Eigen::Vector2d t = T_candidate_query_seed.block<2, 1>(0, 3);
    std::vector<Pair> pairs;
    for (int iteration = 0; iteration < std::max(1, args.refine_iterations); ++iteration) {
        pairs = AssociateOneToOne(
            query, candidate, R, t, args.match_distance, args.dbh_soft_weight
        );
        Eigen::Matrix2d next_R = R;
        Eigen::Vector2d next_t = t;
        if (!EstimateRigid2D(query, candidate, pairs, next_R, next_t)) break;
        R = next_R;
        t = next_t;
    }
    pairs = AssociateOneToOne(
        query, candidate, R, t, args.match_distance, args.dbh_soft_weight
    );
    if (pairs.size() >= 2) {
        EstimateRigid2D(query, candidate, pairs, R, t);
        pairs = AssociateOneToOne(
            query, candidate, R, t, args.match_distance, args.dbh_soft_weight
        );
    }
    out.pairs = static_cast<int>(pairs.size());

    out.query_coverage = query.trees.empty()
        ? 0.0
        : static_cast<double>(pairs.size())
            / static_cast<double>(query.trees.size());

    out.candidate_coverage = candidate.trees.empty()
        ? 0.0
        : static_cast<double>(pairs.size())
            / static_cast<double>(candidate.trees.size());

    const int uni = static_cast<int>(
        query.trees.size() + candidate.trees.size() - pairs.size()
    );

    out.overlap = uni > 0
        ? static_cast<double>(pairs.size()) / uni
        : 0.0;
    if (!pairs.empty()) {
        double residual_sum = 0.0;
        double residual_max = 0.0;
        double dbh_sum = 0.0;
        for (const auto& pair : pairs) {
            const Eigen::Vector2d qp =
                R * Eigen::Vector2d(
                    query.trees[pair.query].x,
                    query.trees[pair.query].y
                ) + t;
            const Eigen::Vector2d cp(
                candidate.trees[pair.candidate].x,
                candidate.trees[pair.candidate].y
            );
            const double residual = (qp - cp).norm();
            residual_sum += residual;
            residual_max = std::max(residual_max, residual);
            dbh_sum += pair.dbh_difference;
        }
        out.mean_residual = residual_sum / pairs.size();
        out.max_residual = residual_max;
        out.mean_dbh_difference = dbh_sum / pairs.size();
    }
    Eigen::Matrix4d T_candidate_query = T_candidate_query_seed;
    T_candidate_query.block<2, 2>(0, 0) = R;
    T_candidate_query.block<2, 1>(0, 3) = t;
    out.T_map_query = treelocpp::PoseToTransform(candidate.pose) * T_candidate_query;
    const Eigen::Vector3d est = out.T_map_query.block<3, 1>(0, 3);
    out.prior_err_xy = std::hypot(
        est.x() - T_map_prior(0, 3),
        est.y() - T_map_prior(1, 3)
    );
    out.prior_err_yaw = std::abs(treelocpp::WrapAngle(
        YawFromMatrix(out.T_map_query.block<3, 3>(0, 0)) - prior_yaw
    ));
    out.valid = pairs.size() >= 2;
    return out;
}

bool Better(const HypothesisResult& a, const HypothesisResult& b) {
    if (a.valid != b.valid) return a.valid;
    if (a.pairs != b.pairs) return a.pairs > b.pairs;

    // For equal pair counts, geometric accuracy is more important than
    // symmetric IoU. IoU otherwise favors sparse database frames.
    if (a.mean_residual != b.mean_residual) {
        return a.mean_residual < b.mean_residual;
    }

    if (a.max_residual != b.max_residual) {
        return a.max_residual < b.max_residual;
    }

    if (std::abs(a.query_coverage - b.query_coverage) > 1e-12) {
        return a.query_coverage > b.query_coverage;
    }

    return a.overlap > b.overlap;
}

struct Row {
    int db_idx = -1;
    double prior_dist = 0.0;
    double retrieval_score = std::numeric_limits<double>::quiet_NaN();
    int hash_score = 0;
    bool rank_transform_ok = false;
    HypothesisResult prior;
    HypothesisResult treeloc;
    HypothesisResult triangle;
    int triangle_hypothesis_count = 0;
    int triangle_consensus_support = 0;
    HypothesisResult selected;
    int accepted = 0;
    std::string reject_reason = "unknown";
};

void ApplyGate(Row& row, const Args& args) {
    const auto& h = row.selected;
    if (!h.valid) row.reject_reason = "hypothesis_not_valid";
    else if (h.pairs < args.min_pairs) row.reject_reason = "few_pairs";
    else if (h.query_coverage < args.min_query_coverage) {
        row.reject_reason = "low_query_coverage";
    } else if (!std::isfinite(h.mean_residual)
               || h.mean_residual > args.max_mean_residual) {
        row.reject_reason = "high_mean_residual";
    } else if (!std::isfinite(h.max_residual)
               || h.max_residual > args.max_max_residual) {
        row.reject_reason = "high_max_residual";
    } else if (h.overlap < args.min_overlap) {
        row.reject_reason = "low_overlap";
    } else if (h.prior_err_xy > args.prior_gate_xy) row.reject_reason = "bad_prior_xy";
    else if (h.prior_err_yaw > args.prior_gate_yaw) row.reject_reason = "bad_prior_yaw";
    else {
        row.accepted = 1;
        row.reject_reason = "ok";
    }
}

bool PassesIntrinsicQuality(const HypothesisResult& hypothesis, const Args& args) {
    return hypothesis.valid
        && hypothesis.pairs >= args.min_pairs
        && hypothesis.query_coverage >= args.min_query_coverage
        && std::isfinite(hypothesis.mean_residual)
        && hypothesis.mean_residual <= args.max_mean_residual
        && std::isfinite(hypothesis.max_residual)
        && hypothesis.max_residual <= args.max_max_residual
        && hypothesis.overlap >= args.min_overlap;
}

double Median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

struct ConsensusResult {
    bool ok = false;
    bool ambiguous = false;
    int support = 0;
    int runner_up_support = 0;
    int representative_db_idx = -1;
    std::vector<int> member_db_indices;
    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
};

ConsensusResult BuildTriangleConsensus(std::vector<Row>& rows, const Args& args) {
    ConsensusResult result;
    std::vector<int> eligible;
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        if (PassesIntrinsicQuality(rows[i].triangle, args)) eligible.push_back(i);
    }
    int best_index = -1;
    for (int i : eligible) {
        const Eigen::Vector3d pi = rows[i].triangle.T_map_query.block<3, 1>(0, 3);
        const double yi = YawFromMatrix(
            rows[i].triangle.T_map_query.block<3, 3>(0, 0)
        );
        int support = 0;
        for (int j : eligible) {
            const Eigen::Vector3d pj = rows[j].triangle.T_map_query.block<3, 1>(0, 3);
            const double yj = YawFromMatrix(
                rows[j].triangle.T_map_query.block<3, 3>(0, 0)
            );
            const double xy = std::hypot(pi.x() - pj.x(), pi.y() - pj.y());
            const double yaw = std::abs(treelocpp::WrapAngle(yi - yj));
            if (xy <= args.consensus_xy && yaw <= args.consensus_yaw) ++support;
        }
        rows[i].triangle_consensus_support = support;
        if (best_index < 0
            || support > rows[best_index].triangle_consensus_support
            || (support == rows[best_index].triangle_consensus_support
                && Better(rows[i].triangle, rows[best_index].triangle))) {
            best_index = i;
        }
    }
    if (best_index < 0) return result;

    const Eigen::Vector3d center =
        rows[best_index].triangle.T_map_query.block<3, 1>(0, 3);
    const double center_yaw = YawFromMatrix(
        rows[best_index].triangle.T_map_query.block<3, 3>(0, 0)
    );
    for (int i : eligible) {
        const Eigen::Vector3d p =
            rows[i].triangle.T_map_query.block<3, 1>(0, 3);
        const double yaw = YawFromMatrix(
            rows[i].triangle.T_map_query.block<3, 3>(0, 0)
        );
        const double xy = std::hypot(center.x() - p.x(), center.y() - p.y());
        const double dyaw = std::abs(treelocpp::WrapAngle(center_yaw - yaw));
        if (xy <= args.consensus_xy && dyaw <= args.consensus_yaw) continue;
        result.runner_up_support = std::max(
            result.runner_up_support,
            rows[i].triangle_consensus_support
        );
    }
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (int i : eligible) {
        const Eigen::Vector3d p = rows[i].triangle.T_map_query.block<3, 1>(0, 3);
        const double yaw = YawFromMatrix(
            rows[i].triangle.T_map_query.block<3, 3>(0, 0)
        );
        const double xy = std::hypot(center.x() - p.x(), center.y() - p.y());
        const double dyaw = std::abs(treelocpp::WrapAngle(center_yaw - yaw));
        if (xy > args.consensus_xy || dyaw > args.consensus_yaw) continue;
        result.member_db_indices.push_back(rows[i].db_idx);
        xs.push_back(p.x());
        ys.push_back(p.y());
        zs.push_back(p.z());
        sin_sum += std::sin(yaw);
        cos_sum += std::cos(yaw);
    }
    result.support = static_cast<int>(result.member_db_indices.size());
    result.representative_db_idx = rows[best_index].db_idx;
    result.ambiguous =
        result.support - result.runner_up_support < args.min_consensus_margin;
    result.ok =
        result.support >= args.min_consensus_support
        && !result.ambiguous;
    if (result.support == 0) return result;
    const double yaw = std::atan2(sin_sum, cos_sum);
    result.T_map_query = PoseFromXYYaw(
        Median(xs), Median(ys), Median(zs), yaw
    );
    return result;
}

}  // namespace

treelocpp::HypothesisLocalizationResult treelocpp::LocalizeHypotheses(
    const FrameData& query,
    const Dataset& database,
    const Pose& map_query_prior,
    const Config& config,
    const HypothesisLocalizationOptions& options) {
    if (query.trees.size() < 3 || query.hashes.empty()) {
        throw std::invalid_argument("query frame is not usable");
    }
    if (database.frames.empty()) {
        throw std::invalid_argument("database dataset has no usable frames");
    }
    if (options.search_radius <= 0.0) {
        throw std::invalid_argument("search_radius must be positive");
    }
    if (options.match_distance <= 0.0) {
        throw std::invalid_argument("match_distance must be positive");
    }
    if (options.refine_iterations < 0) {
        throw std::invalid_argument("refine_iterations must be non-negative");
    }
    if (options.triangle_edge_tolerance <= 0.0) {
        throw std::invalid_argument(
            "triangle_edge_tolerance must be positive");
    }
    if (options.triangle_max_hypotheses <= 0) {
        throw std::invalid_argument(
            "triangle_max_hypotheses must be positive");
    }
    if (options.consensus_xy <= 0.0 ||
        options.consensus_yaw <= 0.0) {
        throw std::invalid_argument(
            "consensus thresholds must be positive");
    }
    if (options.min_consensus_support <= 0) {
        throw std::invalid_argument(
            "min_consensus_support must be positive");
    }
    if (options.min_consensus_margin < 0) {
        throw std::invalid_argument(
            "min_consensus_margin must be non-negative");
    }

    Args args;
    static_cast<HypothesisLocalizationOptions&>(args) = options;
    args.prior_x = map_query_prior.x;
    args.prior_y = map_query_prior.y;
    args.prior_z = map_query_prior.z;

    const Eigen::Matrix4d prior_pose =
        treelocpp::PoseToTransform(map_query_prior);
    args.prior_yaw = YawFromMatrix(
        prior_pose.block<3, 3>(0, 0));

    Dataset query_set;
    query_set.trajectory.push_back(query.pose);
    query_set.frame_to_slot[query.index] = 0;
    query_set.frames.push_back(query);

    const Dataset& database_set = database;

    const auto candidate_slots = CandidateSlots(database_set, args);
    if (candidate_slots.empty()) throw std::runtime_error("no candidate anchors inside search radius");
    const auto ranked = treelocpp::RankCandidates(
        query_set, database_set, 0, candidate_slots, config
    );
    std::unordered_map<int, treelocpp::CandidateResult> rank_by_index;
    for (const auto& result : ranked) rank_by_index[result.candidate_index] = result;
    const Eigen::Matrix4d T_map_prior = PoseFromXYYaw(
        args.prior_x, args.prior_y, args.prior_z, args.prior_yaw
    );
    std::vector<Row> rows;
    for (size_t slot : candidate_slots) {
        const auto& candidate = database_set.frames[slot];
        Row row;
        row.db_idx = candidate.index;
        row.prior_dist = std::hypot(
            candidate.pose.x - args.prior_x,
            candidate.pose.y - args.prior_y
        );
        const Eigen::Matrix4d T_map_candidate = treelocpp::PoseToTransform(candidate.pose);
        const Eigen::Matrix4d T_candidate_query_prior =
            T_map_candidate.inverse() * T_map_prior;
        row.prior = RefineHypothesis(
            query, candidate, T_candidate_query_prior, T_map_prior,
            args.prior_yaw, "prior", args
        );
        row.selected = row.prior;
        auto rank_it = rank_by_index.find(candidate.index);
        if (rank_it != rank_by_index.end()) {
            const auto& rank = rank_it->second;
            row.retrieval_score = rank.retrieval_score;
            row.hash_score = rank.hash_score;
            row.rank_transform_ok = rank.transform.ok;
            if (rank.transform.ok) {
                Eigen::Matrix4d T_candidate_query_treeloc = T_candidate_query_prior;
                T_candidate_query_treeloc.block<2, 2>(0, 0) = rank.transform.R;
                T_candidate_query_treeloc.block<2, 1>(0, 3) = rank.transform.t;
                row.treeloc = RefineHypothesis(
                    query, candidate, T_candidate_query_treeloc, T_map_prior,
                    args.prior_yaw, "treeloc", args
                );
                if (Better(row.treeloc, row.selected)) row.selected = row.treeloc;
            }
        }
        auto triangle_seeds = BuildTolerantTriangleSeeds(query, candidate, args);
        row.triangle_hypothesis_count = static_cast<int>(triangle_seeds.size());
        for (auto& seed : triangle_seeds) {
            seed.T_candidate_query(2, 3) = T_candidate_query_prior(2, 3);
            const auto hypothesis = RefineHypothesis(
                query, candidate, seed.T_candidate_query, T_map_prior,
                args.prior_yaw, "triangle", args
            );
            if (Better(hypothesis, row.triangle)) row.triangle = hypothesis;
        }
        if (Better(row.triangle, row.selected)) row.selected = row.triangle;
        ApplyGate(row, args);
        rows.push_back(std::move(row));
    }
    ConsensusResult consensus = BuildTriangleConsensus(rows, args);
    if (consensus.support > 0) {
        // TreeLoc++ supplies only x/y/yaw. Height remains the external
        // prior (normally LIO/IMU), and roll/pitch are never estimated.
        consensus.T_map_query(2, 3) = args.prior_z;
    }
    const auto is_consensus_member = [&](const Row& row) {
        return std::find(
            consensus.member_db_indices.begin(),
            consensus.member_db_indices.end(),
            row.db_idx
        ) != consensus.member_db_indices.end();
    };
    const auto is_final_selected = [&](const Row& row) {
        return consensus.ok && row.db_idx == consensus.representative_db_idx;
    };
    std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) {
        if (is_final_selected(a) != is_final_selected(b)) {
            return is_final_selected(a);
        }
        if (is_consensus_member(a) != is_consensus_member(b)) {
            return is_consensus_member(a);
        }
        if (a.triangle_consensus_support != b.triangle_consensus_support) {
            return a.triangle_consensus_support > b.triangle_consensus_support;
        }
        if (a.accepted != b.accepted) return a.accepted > b.accepted;
        if (Better(a.selected, b.selected)) return true;
        if (Better(b.selected, a.selected)) return false;
        return a.prior_dist < b.prior_dist;
    });
    const auto representative = std::find_if(
        rows.begin(), rows.end(), [&](const Row& row) {
            return row.db_idx == consensus.representative_db_idx;
        }
    );
    const Eigen::Matrix4d T_odom_query = treelocpp::PoseToTransform(query.pose);
    const double odom_query_yaw = YawFromMatrix(
        T_odom_query.block<3, 3>(0, 0)
    );
    const Eigen::Matrix4d T_odom_query_planar = PoseFromXYYaw(
        query.pose.x, query.pose.y, 0.0, odom_query_yaw
    );
    const Eigen::Matrix4d T_map_prior_planar = PoseFromXYYaw(
        args.prior_x, args.prior_y, 0.0, args.prior_yaw
    );
    const Eigen::Matrix4d T_map_query_planar = PoseFromXYYaw(
        consensus.T_map_query(0, 3),
        consensus.T_map_query(1, 3),
        0.0,
        YawFromMatrix(consensus.T_map_query.block<3, 3>(0, 0))
    );
    const Eigen::Matrix4d T_map_odom_prior =
        T_map_prior_planar * T_odom_query_planar.inverse();
    const Eigen::Matrix4d T_map_odom_candidate =
        T_map_query_planar * T_odom_query_planar.inverse();
    const Eigen::Matrix4d T_difference =
        T_map_odom_candidate * T_map_odom_prior.inverse();
    const PoseSummary map_query_summary = SummarizePose(consensus.T_map_query);
    const PoseSummary odom_query_summary = SummarizePose(T_odom_query);
    const PoseSummary map_odom_summary = SummarizePose(T_map_odom_candidate);
    const PoseSummary map_odom_prior_summary = SummarizePose(T_map_odom_prior);
    const PoseSummary difference_summary = SummarizePose(T_difference);
    const Eigen::Vector3d parameter_translation_difference =
        map_odom_summary.translation - map_odom_prior_summary.translation;
    const double parameter_yaw_difference = treelocpp::WrapAngle(
        map_odom_summary.yaw - map_odom_prior_summary.yaw
    );

    HypothesisLocalizationResult result;
    result.query_index = query.index;
    result.candidate_count = static_cast<int>(candidate_slots.size());
    result.ok = consensus.ok;
    result.ambiguous = consensus.ambiguous;
    result.map_index =
        consensus.ok ? consensus.representative_db_idx : -1;
    result.support = consensus.support;
    result.runner_up_support = consensus.runner_up_support;
    result.consensus_database_indices =
        consensus.member_db_indices;

    if (result.ok) {
        result.status = "ok";
    } else if (result.ambiguous) {
        result.status = "ambiguous";
    } else if (result.support == 0) {
        result.status = "no_consensus";
    } else {
        result.status = "insufficient_consensus";
    }

    const HypothesisResult* representative_hypothesis =
        result.ok && representative != rows.end()
        ? &representative->triangle
        : nullptr;

    if (representative_hypothesis != nullptr) {
        result.pairs = representative_hypothesis->pairs;
        result.overlap = representative_hypothesis->overlap;
        result.query_coverage =
            representative_hypothesis->query_coverage;
        result.candidate_coverage =
            representative_hypothesis->candidate_coverage;
        result.mean_residual =
            representative_hypothesis->mean_residual;
        result.max_residual =
            representative_hypothesis->max_residual;
    }

    if (result.ok) {
        result.T_map_query = consensus.T_map_query;
        result.T_map_odom = T_map_odom_candidate;
        result.T_map_odom_prior = T_map_odom_prior;
        result.lidar_update_planar =
            difference_summary.translation.head<2>().norm();
        result.lidar_update_yaw = difference_summary.yaw;
        result.parameter_planar =
            parameter_translation_difference.head<2>().norm();
        result.parameter_yaw = parameter_yaw_difference;
    }

    result.candidates.reserve(rows.size());
    for (const auto& row : rows) {
        const auto& selected = row.selected;

        HypothesisCandidateDiagnostic diagnostic;
        diagnostic.database_index = row.db_idx;
        diagnostic.prior_distance = row.prior_dist;
        diagnostic.retrieval_score = row.retrieval_score;
        diagnostic.hash_score = row.hash_score;
        diagnostic.rank_transform_ok = row.rank_transform_ok;
        diagnostic.triangle_hypothesis_count =
            row.triangle_hypothesis_count;
        diagnostic.triangle_consensus_support =
            row.triangle_consensus_support;
        diagnostic.intrinsic_ok =
            PassesIntrinsicQuality(row.triangle, args);
        diagnostic.consensus_member =
            is_consensus_member(row);
        diagnostic.final_selected =
            is_final_selected(row);
        diagnostic.accepted = row.accepted != 0;
        diagnostic.reject_reason = row.reject_reason;
        diagnostic.selected_source = selected.source;

        diagnostic.prior_pairs = row.prior.pairs;
        diagnostic.prior_mean_residual =
            row.prior.mean_residual;
        diagnostic.treeloc_pairs = row.treeloc.pairs;
        diagnostic.treeloc_mean_residual =
            row.treeloc.mean_residual;
        diagnostic.triangle_pairs = row.triangle.pairs;
        diagnostic.triangle_mean_residual =
            row.triangle.mean_residual;

        diagnostic.pairs = selected.pairs;
        diagnostic.overlap = selected.overlap;
        diagnostic.query_coverage =
            selected.query_coverage;
        diagnostic.candidate_coverage =
            selected.candidate_coverage;
        diagnostic.mean_residual =
            selected.mean_residual;
        diagnostic.max_residual =
            selected.max_residual;
        diagnostic.mean_dbh_difference =
            selected.mean_dbh_difference;
        diagnostic.prior_error_xy =
            selected.prior_err_xy;
        diagnostic.prior_error_yaw =
            selected.prior_err_yaw;
        diagnostic.T_map_query =
            selected.T_map_query;

        result.candidates.push_back(std::move(diagnostic));
    }

    return result;
}

int treelocpp::RunLocalizeHypothesesCli(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        treelocpp::Config config;
        std::string error;
        if (!treelocpp::LoadConfig(args.config_path, config, &error)) {
            throw std::runtime_error(error);
        }
        treelocpp::RefreshDerivedConfig(config);
        auto query_set = treelocpp::LoadDataset(args.query_root, config, false);
        auto database_set = treelocpp::LoadDataset(
            args.database_root, config, config.neighbor_past_only
        );
        if (query_set.frames.empty()) throw std::runtime_error("query dataset has no usable frames");
        if (database_set.frames.empty()) throw std::runtime_error("database dataset has no usable frames");
        const auto& query = query_set.frames.front();
        const auto candidate_slots = CandidateSlots(database_set, args);
        if (candidate_slots.empty()) throw std::runtime_error("no candidate anchors inside search radius");
        const auto ranked = treelocpp::RankCandidates(
            query_set, database_set, 0, candidate_slots, config
        );
        std::unordered_map<int, treelocpp::CandidateResult> rank_by_index;
        for (const auto& result : ranked) rank_by_index[result.candidate_index] = result;
        const Eigen::Matrix4d T_map_prior = PoseFromXYYaw(
            args.prior_x, args.prior_y, args.prior_z, args.prior_yaw
        );
        std::vector<Row> rows;
        for (size_t slot : candidate_slots) {
            const auto& candidate = database_set.frames[slot];
            Row row;
            row.db_idx = candidate.index;
            row.prior_dist = std::hypot(
                candidate.pose.x - args.prior_x,
                candidate.pose.y - args.prior_y
            );
            const Eigen::Matrix4d T_map_candidate = treelocpp::PoseToTransform(candidate.pose);
            const Eigen::Matrix4d T_candidate_query_prior =
                T_map_candidate.inverse() * T_map_prior;
            row.prior = RefineHypothesis(
                query, candidate, T_candidate_query_prior, T_map_prior,
                args.prior_yaw, "prior", args
            );
            row.selected = row.prior;
            auto rank_it = rank_by_index.find(candidate.index);
            if (rank_it != rank_by_index.end()) {
                const auto& rank = rank_it->second;
                row.retrieval_score = rank.retrieval_score;
                row.hash_score = rank.hash_score;
                row.rank_transform_ok = rank.transform.ok;
                if (rank.transform.ok) {
                    Eigen::Matrix4d T_candidate_query_treeloc = T_candidate_query_prior;
                    T_candidate_query_treeloc.block<2, 2>(0, 0) = rank.transform.R;
                    T_candidate_query_treeloc.block<2, 1>(0, 3) = rank.transform.t;
                    row.treeloc = RefineHypothesis(
                        query, candidate, T_candidate_query_treeloc, T_map_prior,
                        args.prior_yaw, "treeloc", args
                    );
                    if (Better(row.treeloc, row.selected)) row.selected = row.treeloc;
                }
            }
            auto triangle_seeds = BuildTolerantTriangleSeeds(query, candidate, args);
            row.triangle_hypothesis_count = static_cast<int>(triangle_seeds.size());
            for (auto& seed : triangle_seeds) {
                seed.T_candidate_query(2, 3) = T_candidate_query_prior(2, 3);
                const auto hypothesis = RefineHypothesis(
                    query, candidate, seed.T_candidate_query, T_map_prior,
                    args.prior_yaw, "triangle", args
                );
                if (Better(hypothesis, row.triangle)) row.triangle = hypothesis;
            }
            if (Better(row.triangle, row.selected)) row.selected = row.triangle;
            ApplyGate(row, args);
            rows.push_back(std::move(row));
        }
        ConsensusResult consensus = BuildTriangleConsensus(rows, args);
        if (consensus.support > 0) {
            // TreeLoc++ supplies only x/y/yaw. Height remains the external
            // prior (normally LIO/IMU), and roll/pitch are never estimated.
            consensus.T_map_query(2, 3) = args.prior_z;
        }
        const auto is_consensus_member = [&](const Row& row) {
            return std::find(
                consensus.member_db_indices.begin(),
                consensus.member_db_indices.end(),
                row.db_idx
            ) != consensus.member_db_indices.end();
        };
        const auto is_final_selected = [&](const Row& row) {
            return consensus.ok && row.db_idx == consensus.representative_db_idx;
        };
        std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) {
            if (is_final_selected(a) != is_final_selected(b)) {
                return is_final_selected(a);
            }
            if (is_consensus_member(a) != is_consensus_member(b)) {
                return is_consensus_member(a);
            }
            if (a.triangle_consensus_support != b.triangle_consensus_support) {
                return a.triangle_consensus_support > b.triangle_consensus_support;
            }
            if (a.accepted != b.accepted) return a.accepted > b.accepted;
            if (Better(a.selected, b.selected)) return true;
            if (Better(b.selected, a.selected)) return false;
            return a.prior_dist < b.prior_dist;
        });
        const auto representative = std::find_if(
            rows.begin(), rows.end(), [&](const Row& row) {
                return row.db_idx == consensus.representative_db_idx;
            }
        );
        const Eigen::Matrix4d T_odom_query = treelocpp::PoseToTransform(query.pose);
        const double odom_query_yaw = YawFromMatrix(
            T_odom_query.block<3, 3>(0, 0)
        );
        const Eigen::Matrix4d T_odom_query_planar = PoseFromXYYaw(
            query.pose.x, query.pose.y, 0.0, odom_query_yaw
        );
        const Eigen::Matrix4d T_map_prior_planar = PoseFromXYYaw(
            args.prior_x, args.prior_y, 0.0, args.prior_yaw
        );
        const Eigen::Matrix4d T_map_query_planar = PoseFromXYYaw(
            consensus.T_map_query(0, 3),
            consensus.T_map_query(1, 3),
            0.0,
            YawFromMatrix(consensus.T_map_query.block<3, 3>(0, 0))
        );
        const Eigen::Matrix4d T_map_odom_prior =
            T_map_prior_planar * T_odom_query_planar.inverse();
        const Eigen::Matrix4d T_map_odom_candidate =
            T_map_query_planar * T_odom_query_planar.inverse();
        const Eigen::Matrix4d T_difference =
            T_map_odom_candidate * T_map_odom_prior.inverse();
        const PoseSummary map_query_summary = SummarizePose(consensus.T_map_query);
        const PoseSummary odom_query_summary = SummarizePose(T_odom_query);
        const PoseSummary map_odom_summary = SummarizePose(T_map_odom_candidate);
        const PoseSummary map_odom_prior_summary = SummarizePose(T_map_odom_prior);
        const PoseSummary difference_summary = SummarizePose(T_difference);
        const Eigen::Vector3d parameter_translation_difference =
            map_odom_summary.translation - map_odom_prior_summary.translation;
        const double parameter_yaw_difference = treelocpp::WrapAngle(
            map_odom_summary.yaw - map_odom_prior_summary.yaw
        );
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "query_idx=" << query.index
                  << " query_trees=" << query.trees.size()
                  << " candidate_count=" << candidate_slots.size()
                  << " match_distance=" << args.match_distance
                  << " dbh_soft_weight=" << args.dbh_soft_weight
                  << " triangle_edge_tolerance=" << args.triangle_edge_tolerance
                  << " triangle_max_hypotheses=" << args.triangle_max_hypotheses << '\n';
        const Eigen::Vector3d consensus_position =
            consensus.T_map_query.block<3, 1>(0, 3);
        const double consensus_yaw = YawFromMatrix(
            consensus.T_map_query.block<3, 3>(0, 0)
        );
        std::cout << "triangle_consensus_ok=" << (consensus.ok ? 1 : 0)
                  << " support=" << consensus.support
                  << " runner_up_support=" << consensus.runner_up_support
                  << " ambiguous=" << (consensus.ambiguous ? 1 : 0)
                  << " representative_db_idx=" << consensus.representative_db_idx
                  << " est_x=" << consensus_position.x()
                  << " est_y=" << consensus_position.y()
                  << " est_z=" << consensus_position.z()
                  << " est_yaw=" << consensus_yaw
                  << " member_db_indices=";
        for (int i = 0; i < static_cast<int>(consensus.member_db_indices.size()); ++i) {
            if (i > 0) std::cout << ',';
            std::cout << consensus.member_db_indices[i];
        }
        std::cout << '\n';
        const bool localization_ok = consensus.ok;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const HypothesisResult* representative_hypothesis =
            localization_ok && representative != rows.end()
            ? &representative->triangle
            : nullptr;
        const auto result_value = [&](double value) {
            return localization_ok ? value : nan;
        };
        std::cout
            << "localization_result"
            << " ok=" << (localization_ok ? 1 : 0)
            << " query_idx=" << query.index
            << " map_idx="
            << (localization_ok ? consensus.representative_db_idx : -1)
            << " support=" << consensus.support
            << " runner_up_support=" << consensus.runner_up_support
            << " ambiguous=" << (consensus.ambiguous ? 1 : 0)
            << " pairs="
            << (representative_hypothesis ? representative_hypothesis->pairs : 0)
            << " overlap="
            << (representative_hypothesis
                ? representative_hypothesis->overlap
                : std::numeric_limits<double>::quiet_NaN())
            << " mean_residual="
            << (representative_hypothesis
                ? representative_hypothesis->mean_residual
                : std::numeric_limits<double>::quiet_NaN())
            << " max_residual="
            << (representative_hypothesis
                ? representative_hypothesis->max_residual
                : std::numeric_limits<double>::quiet_NaN())
            << " map_c_x=" << result_value(map_query_summary.translation.x())
            << " map_c_y=" << result_value(map_query_summary.translation.y())
            << " map_c_z=" << result_value(map_query_summary.translation.z())
            << " map_c_roll=" << result_value(map_query_summary.roll)
            << " map_c_pitch=" << result_value(map_query_summary.pitch)
            << " map_c_yaw=" << result_value(map_query_summary.yaw)
            << " odom_c_x=" << odom_query_summary.translation.x()
            << " odom_c_y=" << odom_query_summary.translation.y()
            << " odom_c_z=" << odom_query_summary.translation.z()
            << " odom_c_roll=" << odom_query_summary.roll
            << " odom_c_pitch=" << odom_query_summary.pitch
            << " odom_c_yaw=" << odom_query_summary.yaw
            << " map_odom_x=" << result_value(map_odom_summary.translation.x())
            << " map_odom_y=" << result_value(map_odom_summary.translation.y())
            << " map_odom_z=" << result_value(map_odom_summary.translation.z())
            << " map_odom_roll=" << result_value(map_odom_summary.roll)
            << " map_odom_pitch=" << result_value(map_odom_summary.pitch)
            << " map_odom_yaw=" << result_value(map_odom_summary.yaw)
            << " difference_translation_norm="
            << result_value(difference_summary.translation.norm())
            << " difference_planar="
            << result_value(difference_summary.translation.head<2>().norm())
            << " difference_yaw=" << result_value(difference_summary.yaw)
            << " parameter_translation_norm="
            << result_value(parameter_translation_difference.norm())
            << " parameter_planar="
            << result_value(parameter_translation_difference.head<2>().norm())
            << " parameter_yaw=" << result_value(parameter_yaw_difference)
            << '\n';
        std::cout
            << "rank db_idx intrinsic_ok consensus_member final_selected "
            << "reject_reason selected_seed prior_dist "
            << "retrieval_score hash_score rank_transform_ok "
            << "prior_pairs prior_mean_residual treeloc_pairs treeloc_mean_residual "
            << "triangle_hypotheses triangle_pairs triangle_mean_residual "
            << "triangle_consensus_support "
            << "pairs overlap mean_residual max_residual mean_dbh_diff "
            << "est_x est_y est_z est_yaw prior_err_xy prior_err_yaw "
            << "query_coverage candidate_coverage\n";
        const int count = std::min(args.top_k, static_cast<int>(rows.size()));
        for (int i = 0; i < count; ++i) {
            const auto& row = rows[i];
            const auto& h = row.selected;
            const int consensus_member = is_consensus_member(row) ? 1 : 0;
            const int final_selected = is_final_selected(row) ? 1 : 0;
            const Eigen::Vector3d p = h.T_map_query.block<3, 1>(0, 3);
            const double yaw = YawFromMatrix(h.T_map_query.block<3, 3>(0, 0));
            std::cout << i << ' ' << row.db_idx << ' ' << row.accepted << ' '
                      << consensus_member << ' ' << final_selected << ' '
                      << row.reject_reason << ' ' << h.source << ' '
                      << row.prior_dist << ' ' << row.retrieval_score << ' '
                      << row.hash_score << ' ' << (row.rank_transform_ok ? 1 : 0) << ' '
                      << row.prior.pairs << ' ' << row.prior.mean_residual << ' '
                      << row.treeloc.pairs << ' ' << row.treeloc.mean_residual << ' '
                      << row.triangle_hypothesis_count << ' '
                      << row.triangle.pairs << ' ' << row.triangle.mean_residual << ' '
                      << row.triangle_consensus_support << ' '
                      << h.pairs << ' ' << h.overlap << ' '
                      << h.mean_residual << ' ' << h.max_residual << ' '
                      << h.mean_dbh_difference << ' '
                      << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << yaw << ' '
                      << h.prior_err_xy << ' ' << h.prior_err_yaw << ' '
                      << h.query_coverage << ' '
                      << h.candidate_coverage << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "treelocpp_localize_hypotheses error: " << e.what() << '\n';
        return 1;
    }
}
