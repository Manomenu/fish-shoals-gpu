#pragma once
#include "config.cuh"
#include "cube.h"
#include "shader.h"

struct RectangleModelCreateInfo {
	glm::vec3 size;
};

class RectangleModel {
public:
	unsigned int VBO, VAO, vertexCount;
	std::vector<float> vertices;

	RectangleModel(RectangleModelCreateInfo* createInfo);
	~RectangleModel();
	void render(Cube* cube, Shader* shader, bool shouldFill, 
		const glm::mat4& view, const glm::mat4& projection);
};

