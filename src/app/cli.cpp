#include "cli.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>

static bool has_ext(const std::string& s, const char* ext) {
	return s.size() >= 4 && s.compare(s.size() - 4, 4, ext) == 0;
}

static void print_usage(const char* prog) {
	std::cerr <<
	"Usage: " << prog << " <output[.ppm|.exr]> [scene] [options]\n"
	"\n"
	"Scenes (default: cornell)\n"
	"  cornell         Cornell box with two smoke volumes\n"
	"  furnace         white furnace test; energy conservation\n"
	"  closed_furnace  closed box, where L = Le/(1-rho) must converge to 1\n"
	"  ggx             roughness and metalness sweep\n"
	"  hdr             the same sweep under an environment map\n"
	"  glass           rough dielectric spheres\n"
	"  caustics        glass and metal in a Cornell box; defaults to BDPT\n"
	"  sss             subsurface scattering\n"
	"  volume          participating media\n"
	"  ppm             specular-to-diffuse transport; defaults to photon mapping\n"
	"  bunny           OBJ mesh (models/bunny.obj)\n"
	"  helmet          glTF mesh (models/helmet)\n"
	"\n"
	"Options\n"
	"  -w, --width N            image width           (default 400)\n"
	"  -h, --height N           image height          (default 400)\n"
	"  -s, --spp N              samples per pixel     (default 64)\n"
	"  -d, --depth N            maximum bounce depth  (default 10)\n"
	"  -t, --tile N             tile size, CPU only   (default 32)\n"
	"      --device cpu|gpu     render backend        (default cpu)\n"
	"      --integrator pt|bdpt|ppm|restir   override the scene's default\n"
	"      --denoise            run Open Image Denoise on the result\n"
	"      --no-preview         headless; no OpenGL window\n"
	"      --viewport           open the scene in the viewport instead\n"
	"      --rendered           open the viewport straight into rendered mode\n"
	"      --help               this message\n"
	"\n"
	"The GPU implements cornell, furnace, ggx, hdr, bunny, glass, caustics,\n"
	"volume and sss; anything else falls back to the CPU with a note.\n"
	"Photon mapping is CPU only. An .exr extension writes OpenEXR.\n"
	"Viewport controls are in the README.\n";
}

RenderConfig parse_cli(int argc, char** argv)
{
	RenderConfig config;

	// Printing usage and then rendering a default scene anyway is not what
	// anyone who ran this with no arguments was asking for.
	if (argc < 2) { print_usage(argv[0]); std::exit(1); }

	for (int i = 1; i < argc; ++i)
		if (std::string(argv[i]) == "--help") {
			print_usage(argv[0]);
			std::exit(0);
		}

	config.output_path = argv[1];
	if (argc >= 3 && argv[2][0] != '-')
		config.feature = argv[2];

	// Presets are defaults, so they go in before the flag loop below and an
	// explicit --width/--spp/--depth still wins.
	if (config.feature == "furnace")
		apply_furnace_preset(config);

	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		if      ((a == "--width"  || a == "-w") && i+1 < argc) config.width     = std::atoi(argv[++i]);
		else if ((a == "--height" || a == "-h") && i+1 < argc) config.height    = std::atoi(argv[++i]);
		else if ((a == "--spp"    || a == "-s") && i+1 < argc) config.samples   = std::atoi(argv[++i]);
		else if ((a == "--depth"  || a == "-d") && i+1 < argc) config.max_depth = std::atoi(argv[++i]);
		else if ((a == "--tile"   || a == "-t") && i+1 < argc) config.tile_size = std::atoi(argv[++i]);
	}

	// main() picks the writer off this extension, so .exr has to survive here.
	if (!has_ext(config.output_path, ".ppm") &&
		!has_ext(config.output_path, ".exr"))
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