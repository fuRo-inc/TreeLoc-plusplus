#pragma once

#include <vector>

#include "treelocpp/config.h"
#include "treelocpp/types.h"

namespace treelocpp {

Transform2D EstimateTransform2D(const FrameData& query,
                                const FrameData& candidate,
                                const Config& config);
PoseCorrection EstimateVerticalCorrection(const FrameData& query,
                                          const FrameData& candidate,
                                          const Transform2D& transform,
                                          const Config& config);
std::vector<CandidateResult> RankCandidates(const Dataset& query_set,
                                            const Dataset& database_set,
                                            size_t query_slot,
                                            const std::vector<size_t>& candidate_slots,
                                            const Config& config);

}  // namespace treelocpp
