#include "cli.h"

#include <filesystem>
#include <iostream>

RenderConfig parse_cli(int argc, char** argv)
{
    RenderConfig config;

    if (argc >= 2)
        config.output_path = argv[1];

    if (argc >= 3)
        config.feature = argv[2];

    if (config.output_path.size() < 4 ||
        config.output_path.substr(config.output_path.size() - 4) != ".ppm")
    {
        config.output_path += ".ppm";
    }

    std::filesystem::create_directories("renders");

    config.output_path =
        (std::filesystem::current_path() /
         "renders" /
         config.output_path).string();

    return config;
}