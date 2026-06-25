#include "treelocpp/config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace treelocpp {

namespace {

std::string Trim(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string StripComment(const std::string& line) {
    bool single = false;
    bool dbl = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\'' && !dbl) single = !single;
        if (line[i] == '"' && !single) dbl = !dbl;
        if (line[i] == '#' && !single && !dbl) return line.substr(0, i);
    }
    return line;
}

std::string ParseString(const std::string& value) {
    std::string out = Trim(value);
    if (out.size() >= 2 &&
        ((out.front() == '"' && out.back() == '"') ||
         (out.front() == '\'' && out.back() == '\''))) {
        out = out.substr(1, out.size() - 2);
    }
    return out;
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

template <typename T>
T ParseNumber(const std::string& value, const std::string& key, size_t line_no) {
    std::stringstream ss(value);
    T parsed;
    ss >> parsed;
    if (!ss || !ss.eof()) {
        throw std::runtime_error("invalid value for " + key + " on line " +
                                 std::to_string(line_no));
    }
    return parsed;
}

bool ParseBool(const std::string& value, const std::string& key, size_t line_no) {
    const std::string v = Lower(ParseString(value));
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    throw std::runtime_error("invalid boolean for " + key + " on line " +
                             std::to_string(line_no));
}

void Assign(const std::string& key, const std::string& value, Config& config, size_t line_no) {
    if (key == "mode") config.mode = ParseString(value);
    else if (key == "dataset_root") config.dataset_root = ParseString(value);
    else if (key == "query_root") config.query_root = ParseString(value);
    else if (key == "database_root") config.database_root = ParseString(value);
    else if (key == "max_frames") config.max_frames = ParseNumber<int>(value, key, line_no);
    else if (key == "spatial_threshold") config.spatial_threshold = ParseNumber<double>(value, key, line_no);
    else if (key == "use_test_polygons") config.use_test_polygons = ParseBool(value, key, line_no);
    else if (key == "temporal_min_separation") config.temporal_min_separation = ParseNumber<int>(value, key, line_no);
    else if (key == "recall_k") config.recall_k = ParseNumber<int>(value, key, line_no);
    else if (key == "histogram_k") config.histogram_k = ParseNumber<int>(value, key, line_no);
    else if (key == "rerank_k") config.rerank_k = ParseNumber<int>(value, key, line_no);
    else if (key == "knn_k") config.knn_k = ParseNumber<int>(value, key, line_no);
    else if (key == "min_dist") config.min_dist = ParseNumber<double>(value, key, line_no);
    else if (key == "max_dist") config.max_dist = ParseNumber<double>(value, key, line_no);
    else if (key == "delta_l") config.delta_l = ParseNumber<double>(value, key, line_no);
    else if (key == "rho") config.rho = ParseNumber<long long>(value, key, line_no);
    else if (key == "hash_modulus") config.hash_modulus = ParseNumber<long long>(value, key, line_no);
    else if (key == "number_of_cluster") config.number_of_cluster = ParseNumber<int>(value, key, line_no);
    else if (key == "local_radius") config.local_radius = ParseNumber<double>(value, key, line_no);
    else if (key == "tree_score_min") config.tree_score_min = ParseNumber<double>(value, key, line_no);
    else if (key == "neighbor_augment") config.neighbor_augment = ParseBool(value, key, line_no);
    else if (key == "neighbor_past_only") config.neighbor_past_only = ParseBool(value, key, line_no);
    else if (key == "neighbor_max_scenes") config.neighbor_max_scenes = ParseNumber<int>(value, key, line_no);
    else if (key == "neighbor_radius") config.neighbor_radius = ParseNumber<double>(value, key, line_no);
    else if (key == "min_reconstructed_per_frame") config.min_reconstructed_per_frame = ParseNumber<int>(value, key, line_no);
    else if (key == "dedup_distance") config.dedup_distance = ParseNumber<double>(value, key, line_no);
    else if (key == "apply_axis_alignment") config.apply_axis_alignment = ParseBool(value, key, line_no);
    else if (key == "dataset_yaw_deg") config.dataset_yaw_deg = ParseNumber<double>(value, key, line_no);
    else if (key == "query_yaw_deg") config.query_yaw_deg = ParseNumber<double>(value, key, line_no);
    else if (key == "database_yaw_deg") config.database_yaw_deg = ParseNumber<double>(value, key, line_no);
    else if (key == "min_radius") config.min_radius = ParseNumber<double>(value, key, line_no);
    else if (key == "max_radius") config.max_radius = ParseNumber<double>(value, key, line_no);
    else if (key == "total_section") config.total_section = ParseNumber<int>(value, key, line_no);
    else if (key == "bin_width") config.bin_width = ParseNumber<double>(value, key, line_no);
    else if (key == "spatial_bin_interval") config.spatial_bin_interval = ParseNumber<double>(value, key, line_no);
    else if (key == "spatial_bin_padding") config.spatial_bin_padding = ParseNumber<double>(value, key, line_no);
    else if (key == "spatial_bin_count") config.spatial_bin_count = ParseNumber<int>(value, key, line_no);
    else if (key == "spatial_bin_min") config.spatial_bin_min = ParseNumber<double>(value, key, line_no);
    else if (key == "spatial_bin_max") config.spatial_bin_max = ParseNumber<double>(value, key, line_no);
    else if (key == "tdh_use_rec_only") config.tdh_use_rec_only = ParseBool(value, key, line_no);
    else if (key == "pairwise_use_rec_only") config.pairwise_use_rec_only = ParseBool(value, key, line_no);
    else if (key == "pairwise_weight") config.pairwise_weight = ParseNumber<double>(value, key, line_no);
    else if (key == "pairwise_min_dist") config.pairwise_min_dist = ParseNumber<double>(value, key, line_no);
    else if (key == "pairwise_max_dist") config.pairwise_max_dist = ParseNumber<double>(value, key, line_no);
    else if (key == "pairwise_bins") config.pairwise_bins = ParseNumber<int>(value, key, line_no);
    else if (key == "pairwise_max_pairs") config.pairwise_max_pairs = ParseNumber<int>(value, key, line_no);
    else if (key == "pairwise_soft_binning") config.pairwise_soft_binning = ParseBool(value, key, line_no);
    else if (key == "use_t_aware_overlap") config.use_t_aware_overlap = ParseBool(value, key, line_no);
    else if (key == "t_aware_tau") config.t_aware_tau = ParseNumber<double>(value, key, line_no);
    else if (key == "t_aware_power") config.t_aware_power = ParseNumber<double>(value, key, line_no);
    else if (key == "t_aware_mode") config.t_aware_mode = ParseString(value);
    else if (key == "use_yaw_voting") config.use_yaw_voting = ParseBool(value, key, line_no);
    else if (key == "yaw_bin_deg") config.yaw_bin_deg = ParseNumber<double>(value, key, line_no);
    else if (key == "yaw_inlier_tol_deg") config.yaw_inlier_tol_deg = ParseNumber<double>(value, key, line_no);
    else if (key == "yaw_min_tri_inliers") config.yaw_min_tri_inliers = ParseNumber<int>(value, key, line_no);
    else if (key == "use_dbh_triangle_match") config.use_dbh_triangle_match = ParseBool(value, key, line_no);
    else if (key == "dbh_diff_tol") config.dbh_diff_tol = ParseNumber<double>(value, key, line_no);
    else if (key == "match_distance_tol") config.match_distance_tol = ParseNumber<double>(value, key, line_no);
    else if (key == "use_vertical") config.use_vertical = ParseBool(value, key, line_no);
    else if (key == "vertical_ransac_iters") config.vertical_ransac_iters = ParseNumber<int>(value, key, line_no);
    else if (key == "vertical_min_sample") config.vertical_min_sample = ParseNumber<int>(value, key, line_no);
    else if (key == "z_inlier_tol") config.z_inlier_tol = ParseNumber<double>(value, key, line_no);
    else if (key == "z_inlier_ratio") config.z_inlier_ratio = ParseNumber<double>(value, key, line_no);
    else throw std::runtime_error("unknown config key " + key + " on line " + std::to_string(line_no));
}

}  // namespace

std::filesystem::path DefaultConfigPath(const std::string& mode) {
    return std::filesystem::path("config") / (mode == "inter" ? "inter.yaml" : "default.yaml");
}

bool LoadConfig(const std::filesystem::path& path, Config& config, std::string* error) {
    try {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("could not open config: " + path.string());
        std::string line;
        size_t line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const std::string trimmed = Trim(StripComment(line));
            if (trimmed.empty()) continue;
            const size_t colon = trimmed.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error("expected key: value on line " + std::to_string(line_no));
            }
            Assign(Trim(trimmed.substr(0, colon)), Trim(trimmed.substr(colon + 1)), config, line_no);
        }
        RefreshDerivedConfig(config);
        return ValidateConfig(config, error);
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return false;
    }
}

void RefreshDerivedConfig(Config& config) {
    config.spatial_range_bins.clear();
    for (int i = 0; i < config.spatial_bin_count; ++i) {
        const double start0 = config.spatial_bin_min + i * config.spatial_bin_interval;
        const double end0 = start0 + config.spatial_bin_interval;
        const double lo = std::max(config.spatial_bin_min, start0 - config.spatial_bin_padding);
        const double padded_end = (i == 0) ? end0 : end0 + config.spatial_bin_padding;
        const double hi = std::min(config.spatial_bin_max, padded_end);
        if (hi > lo) config.spatial_range_bins.emplace_back(lo, hi);
    }
}

bool ValidateConfig(const Config& config, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (config.max_frames < 0) return fail("max_frames must be non-negative");
    if (config.spatial_threshold <= 0.0) return fail("spatial_threshold must be positive");
    if (config.histogram_k <= 0 || config.rerank_k <= 0) return fail("candidate counts must be positive");
    if (config.knn_k <= 0) return fail("knn_k must be positive");
    if (config.max_dist <= config.min_dist) return fail("max_dist must be greater than min_dist");
    if (config.delta_l <= 0.0) return fail("delta_l must be positive");
    if (config.number_of_cluster <= 0) return fail("number_of_cluster must be positive");
    if (config.local_radius <= 0.0) return fail("local_radius must be positive");
    if (config.total_section <= 0 || config.bin_width <= 0.0) return fail("radius bins are invalid");
    if (config.pairwise_bins <= 0) return fail("pairwise_bins must be positive");
    if (config.pairwise_max_dist <= config.pairwise_min_dist) return fail("pairwise distance range is invalid");
    if (config.spatial_range_bins.empty()) return fail("spatial bins are empty");
    return true;
}

std::vector<RangeBin> BuildRadiusBins(const Config& config) {
    std::vector<RangeBin> bins;
    if (config.total_section == 1) {
        bins.emplace_back(0.0, std::numeric_limits<double>::infinity());
        return bins;
    }
    const double step = (config.max_radius - config.min_radius) / (config.total_section + 1);
    for (int i = 0; i < config.total_section - 1; ++i) {
        const double lo = config.min_radius + i * step;
        bins.emplace_back(lo, lo + config.bin_width);
    }
    bins.emplace_back(bins.back().first + step, std::numeric_limits<double>::infinity());
    return bins;
}

}  // namespace treelocpp
