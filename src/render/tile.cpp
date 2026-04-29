#include "render/tile.h"
#include <algorithm>

std::vector<Tile> generate_tiles(
	int width,
	int height,
	int tile_size
) {
	std::vector<Tile> tiles;

	for (int y = 0; y < height; y += tile_size) {
		for (int x = 0; x < width; x += tile_size) {

			Tile t;
			t.x0 = x;
			t.y0 = y;
			t.x1 = std::min(x + tile_size, width);
			t.y1 = std::min(y + tile_size, height);

			tiles.push_back(t);
		}
	}

	return tiles;
}