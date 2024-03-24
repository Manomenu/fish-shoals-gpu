#pragma once
#include "config.cuh"
#include "shader.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "fish_model.cuh"
#include "fishes.cuh"

struct FishesModelCreateInfo {
	int numberOfFishes;
};

class FishesModel
{
public:
	unsigned int VBO, VAO, numberOfFishes, numberOfBlocks;
	glm::mat4 models[FISH_COUNT];
	FishModel* fish;

	// accessible on a gpu only
	struct cudaGraphicsResource* resource_vbo;
	glm::mat4* dev_models;


	FishesModel(FishesModelCreateInfo* createInfo);
	~FishesModel();
	void render(Shader* shader, Fishes* fishes,
		const glm::mat4& view, const glm::mat4& projection);
	void setModels(
		glm::mat4* models, 
		std::vector<glm::vec3>& positions, 
		std::vector<glm::vec3>& velocities
	);
};
