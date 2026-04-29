#pragma once
#include <vector>

struct Tile {
	int x0, y0;
	int x1, y1;
};

std::vector<Tile> generate_tiles(
	int width,
	int height,
	int tile_size
);