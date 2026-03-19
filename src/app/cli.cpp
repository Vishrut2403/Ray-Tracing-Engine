#include "cli.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <output> [scene]"
              << " [--width N] [--height N] [--spp N] [--depth N] [--tile N]\n"
              << "  scene: cornell | furnace  (default: cornell)\n";
}

RenderConfig parse_cli(int argc, char** argv)
{
    RenderConfig config;

    if (argc < 2) { print_usage(argv[0]); return config; }

    config.output_path = argv[1];
    if (argc >= 3 && argv[2][0] != '-')
        config.feature = argv[2];

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      ((a == "--width"  || a == "-w") && i+1 < argc) config.width     = std::atoi(argv[++i]);
        else if ((a == "--height" || a == "-h") && i+1 < argc) config.height    = std::atoi(argv[++i]);
        else if ((a == "--spp"    || a == "-s") && i+1 < argc) config.samples   = std::atoi(argv[++i]);
        else if ((a == "--depth"  || a == "-d") && i+1 < argc) config.max_depth = std::atoi(argv[++i]);
        else if ((a == "--tile"   || a == "-t") && i+1 < argc) config.tile_size = std::atoi(argv[++i]);
    }

    if (config.output_path.size() < 4 ||
        config.output_path.substr(config.output_path.size() - 4) != ".ppm")
        config.output_path += ".ppm";

    namespace fs = std::filesystem;

    fs::path out(config.output_path);

    if (out.has_parent_path() && out.parent_path() != fs::path(".")) {
        // Path includes a directory (e.g. results/out.ppm) — use as-is
        fs::create_directories(out.parent_path());
        config.output_path = (fs::current_path() / out).string();
    } else {
        // Bare filename — put it in renders/ as before
        fs::create_directories("renders");
        config.output_path = (fs::current_path() / "renders" / out).string();
    }

    return config;
}