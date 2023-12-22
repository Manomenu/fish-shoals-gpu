#include "cube.h"

Cube::Cube(CubeCreateInfo* createInfo) {
	position = createInfo->position;
	size = createInfo->size;
}

Cube::Cube() {
	position = glm::vec3(0);
	size = glm::vec3(0);
}
