#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_USE_CPP14
#include "external/tiny_gltf.h"

namespace tinygltf {
bool WriteImageData(const std::string*, const std::string*,
					const Image*, bool,
					const FsCallbacks*, const URICallbacks*,
					std::string*, void*) {
	return false;
}
}