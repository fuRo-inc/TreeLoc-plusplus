#pragma once

namespace treelocpp {

// Compatibility entry point used by the existing command-line application.
// The in-memory localization API will be added separately.
int RunLocalizeHypothesesCli(int argc, char** argv);

}  // namespace treelocpp
