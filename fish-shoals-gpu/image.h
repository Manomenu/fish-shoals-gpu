#pragma once
#include "config.cuh"

namespace util {
	image load_from_file(const char* filename);

	void free_image_memory(image oldImage);
}

