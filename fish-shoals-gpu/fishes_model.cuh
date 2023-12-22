#include "config.cuh"
#include "shader.h"
#include "fish_transformation.h"
#include "pyramid_model.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

struct FishesModelCreateInfo {
	int numberOfFishes;
};

class FishesModel
{
public:
	unsigned int VBO, VAO, EBO, vertexCount, numberOfFishes;
	std::vector<float> vertices;

	FishesModel(FishesModelCreateInfo* createInfo);
	~FishesModel();
	void render(Shader* shader, FishTransformation* fishes);

private:
	PyramidModel* fishModel;
	cudaGraphicsResource* resource_vbo;
	size_t vbo_size;
	float* dev_vbo_data; // accessible on a gpu only
	float* dev_fishes_positions;
	PyramidModel* dev_fish_model;

	__global__ void setFishesVertices(float* dev_vbo_data, float* fishes_positions, PyramidModel* fishModel);
};
