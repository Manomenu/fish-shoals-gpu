#pragma once
#include "config.cuh"
#include "cube.h"
#include "shader.h"
#include "material.h"

struct RectangleModelCreateInfo {
	glm::vec3 size;
};

class RectangleModel {
public:
	unsigned int VBO, VAO, vertexCount;
	std::vector<float> vertices;

	RectangleModel(RectangleModelCreateInfo* createInfo);
	~RectangleModel();
	void render(Cube* cube, Material* material, Shader* shader, bool shouldFill);
};

