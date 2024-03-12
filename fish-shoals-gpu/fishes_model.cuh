#include "config.cuh"
#include "shader.h"
#include "fish_transformation.h"
#include "fish_physics.h"
#include "pyramid_model.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

//__global__ void setFishesVertices(float* dev_vbo_data, float* fishes_positions, PyramidModel* fishModel)
//{
//	int tid = threadIdx.x + blockDim.x * blockIdx.x;
//
//	dev_vbo_data[tid] = tid;
//}

struct FishesModelCreateInfo {
	int numberOfFishes;
};

class FishesModel
{
public:
	unsigned int VBO, VAO, EBO, vertexCount, numberOfFishes, numberOfBlocks;
	std::vector<float> vertices;

	FishesModel(FishesModelCreateInfo* createInfo);
	~FishesModel();
	void render(Shader* shader, float* dev_fishes_positions, 
		float* dev_fishes_directions,
		const glm::mat4& view, const glm::mat4& projection);

private:
	float h = 0.5f;
	float a = 0.3f;

	float fishModel[15] = {
		-a, -h, a, //0
		a, -h, a, //1
		a, -h, -a, //2
		-a, -h, -a, //3
		0.0, h, 0.0 //4
	};
	struct cudaGraphicsResource* resource_vbo;

	// accessible on a gpu only
	float3* dev_vbo_data; 
	// float* dev_fishes_positions;
	float* dev_fish_model;
};
