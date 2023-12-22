#pragma once
#include "config.cuh"
#include "shader.h"

struct PyramidModelCreateInfo {
	glm::vec2 size;
};

class PyramidModel {
public:
	unsigned int /*VBO, VAO,*/ vertexCount;
	std::vector<float> vertices;

	PyramidModel(PyramidModelCreateInfo* createInfo);
	~PyramidModel();
	//void render(FishTransformation* fish, Shader* shader);
};

