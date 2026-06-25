#include <iostream>

#include "treelocpp/eval.h"

int main(int argc, char** argv) {
    treelocpp::Config config;
    std::string error;
    const std::filesystem::path config_path =
        argc > 1 ? argv[1] : treelocpp::DefaultConfigPath("inter");
    if (!treelocpp::LoadConfig(config_path, config, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    config.mode = "inter";
    const auto summary = treelocpp::RunInterSession(config);
    treelocpp::PrintSummary(summary, std::cout);
    return 0;
}
