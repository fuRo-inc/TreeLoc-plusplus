#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "treelocpp/config.h"
#include "treelocpp/descriptors.h"
#include "treelocpp/types.h"

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

treelocpp::Tree MakeTree(double x,
                        double y,
                        int reconstructed,
                        int number_clusters,
                        double score) {
    treelocpp::Tree tree;
    tree.x = x;
    tree.y = y;
    tree.reconstructed = reconstructed;
    tree.number_clusters = number_clusters;
    tree.score = score;
    return tree;
}

}  // namespace

int main() {
    try {
        treelocpp::Config config;
        config.number_of_cluster = 3;
        config.local_radius = 10.0;
        config.tree_score_min = 0.1;
        config.dedup_distance = 0.2;
        config.tree_axis_alignment_enabled = false;
        config.tdh_use_rec_only = false;
        config.pdh_use_rec_only = false;
        config.knn_k = 3;
        config.min_dist = 0.1;
        config.max_dist = 10.0;
        treelocpp::RefreshDerivedConfig(config);

        treelocpp::Pose pose;
        pose.stamp = 12.5;
        pose.x = 1.0;
        pose.y = -2.0;
        pose.z = 3.0;

        std::vector<treelocpp::Tree> trees;
        trees.push_back(MakeTree(0.0, 0.0, 1, 3, 0.5));
        trees.push_back(MakeTree(2.0, 0.0, 0, 3, 0.7));
        trees.push_back(MakeTree(0.0, 2.0, 0, 2, 0.9));
        trees.push_back(MakeTree(11.0, 0.0, 1, 3, 1.0));
        trees.push_back(MakeTree(1.0, 1.0, 0, 4, 0.05));

        const treelocpp::FrameData frame =
            treelocpp::BuildFrameData(42, pose, trees, config);

        Require(frame.index == 42, "frame index was not preserved");
        Require(std::abs(frame.pose.stamp - 12.5) < 1.0e-12,
                "pose timestamp was not preserved");
        Require(frame.trees.size() == 3, "unexpected selected tree count");
        Require(frame.centers.size() == 3, "unexpected center count");
        Require(frame.trees[0].reconstructed == 1,
                "reconstructed tree must be selected first");
        Require(frame.trees[1].number_clusters > 2,
                "partial tree with more than two clusters must be preferred");
        Require(frame.trees[2].number_clusters == 2,
                "two-cluster partial tree must be selected next");
        Require(frame.alignment_transform.isApprox(
                    Eigen::Matrix4d::Identity(), 1.0e-12),
                "alignment must remain identity when disabled");
        Require(!frame.triangles.simplices.empty(),
                "triangle descriptors were not generated");
        Require(!frame.hashes.empty(), "triangle hashes were not generated");
        Require(!frame.hash_counts.empty(),
                "triangle hash counts were not generated");

        std::cout << "treelocpp frame data API smoke test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "treelocpp frame data API smoke test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
