#include "treelocpp/io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace treelocpp {

namespace {

std::vector<std::string> SplitCsv(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (!line.empty() && line.back() == ',') cells.emplace_back();
    return cells;
}

double ToDouble(const std::string& value, double fallback) {
    if (value.empty()) return fallback;
    try {
        return std::stod(value);
    } catch (...) {
        return fallback;
    }
}

int ToInt(const std::string& value, int fallback) {
    if (value.empty()) return fallback;
    if (value == "True" || value == "true") return 1;
    if (value == "False" || value == "false") return 0;
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

std::string Cell(const std::vector<std::string>& cells,
                 const std::unordered_map<std::string, size_t>& columns,
                 const std::string& name) {
    auto it = columns.find(name);
    if (it == columns.end() || it->second >= cells.size()) return "";
    return cells[it->second];
}

}  // namespace

std::vector<Pose> ReadTrajectory(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::vector<Pose> poses;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        Pose p;
        ss >> p.stamp >> p.x >> p.y >> p.z >> p.qx >> p.qy >> p.qz >> p.qw;
        if (!ss.fail()) poses.push_back(p);
    }
    return poses;
}

std::vector<Tree> ReadTreeCsv(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::vector<Tree> trees;
    if (!in) return trees;

    std::string line;
    if (!std::getline(in, line)) return trees;
    const auto header = SplitCsv(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;

    bool has_axis = true;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!columns.count("axis_" + std::to_string(r) + std::to_string(c))) {
                has_axis = false;
            }
        }
    }

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cells = SplitCsv(line);
        Tree tree;
        tree.has_axis = has_axis;
        if (has_axis) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    tree.axis(r, c) =
                        ToDouble(Cell(cells, columns, "axis_" + std::to_string(r) + std::to_string(c)),
                                 r == c ? 1.0 : 0.0);
                }
            }
        }
        tree.x = ToDouble(Cell(cells, columns, "location_x"), std::numeric_limits<double>::quiet_NaN());
        tree.y = ToDouble(Cell(cells, columns, "location_y"), std::numeric_limits<double>::quiet_NaN());
        tree.z = ToDouble(Cell(cells, columns, "location_z"), std::numeric_limits<double>::quiet_NaN());
        tree.alignment_z = ToDouble(Cell(cells, columns, "alignment_z"), tree.z);
        tree.dbh = ToDouble(Cell(cells, columns, "dbh"), std::numeric_limits<double>::quiet_NaN());
        tree.dbh_approximation =
            ToDouble(Cell(cells, columns, "dbh_approximation"), std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(tree.dbh) && std::isfinite(tree.dbh_approximation)) tree.dbh = tree.dbh_approximation;
        if (!std::isfinite(tree.dbh_approximation) && std::isfinite(tree.dbh)) tree.dbh_approximation = tree.dbh;
        tree.score = ToDouble(Cell(cells, columns, "score"),
                              ToDouble(Cell(cells, columns, "scores"), 1.0));
        tree.reconstructed = ToInt(Cell(cells, columns, "reconstructed"), 1);
        tree.number_clusters = ToInt(Cell(cells, columns, "number_clusters"), 3);
        if (std::isfinite(tree.x) && std::isfinite(tree.y) && std::isfinite(tree.dbh)) {
            trees.push_back(tree);
        }
    }
    return trees;
}

bool HasFrameCsv(const std::filesystem::path& root, int index) {
    return std::filesystem::exists(root / ("TreeManagerState_" + std::to_string(index) + ".csv"));
}

std::vector<int> DiscoverFrameIndices(const std::filesystem::path& root, int max_frames) {
    std::vector<int> indices;
    if (max_frames <= 0) {
        const std::string prefix = "TreeManagerState_";
        const std::string suffix = ".csv";
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size() + suffix.size()) continue;
            if (name.substr(name.size() - suffix.size()) != suffix) continue;
            const std::string id = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
            if (!std::all_of(id.begin(), id.end(), [](unsigned char ch) { return std::isdigit(ch); })) continue;
            indices.push_back(std::stoi(id));
        }
        std::sort(indices.begin(), indices.end());
        return indices;
    }
    for (int i = 0; i < max_frames; ++i) {
        if (HasFrameCsv(root, i)) indices.push_back(i);
    }
    return indices;
}

std::string DatasetName(const std::filesystem::path& root) {
    return root.filename().empty() ? root.string() : root.filename().string();
}

}  // namespace treelocpp
