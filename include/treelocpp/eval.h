#pragma once

#include <iosfwd>

#include "treelocpp/config.h"
#include "treelocpp/types.h"

namespace treelocpp {

EvaluationSummary RunIntraSession(const Config& config);
EvaluationSummary RunInterSession(const Config& config);
void PrintSummary(const EvaluationSummary& summary, std::ostream& out);

}  // namespace treelocpp
