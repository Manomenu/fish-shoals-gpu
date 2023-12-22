#pragma once
#include "config.cuh"

struct CubeCreateInfo {
	glm::vec3 position;
	glm::vec3 size;
};

class Cube 
{
public:
	glm::vec3 position;
	glm::vec3 size;

	Cube(CubeCreateInfo* createInfo);

protected:
	Cube();
};

