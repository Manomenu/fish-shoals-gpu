#pragma once
#include "config.cuh"
#include "device_launch_parameters.h"

struct CreateFishesInfo {
	int numberOfFishes;
	glm::vec3 aquariumSize;
};

struct fishesParams {
	/*const float MIN_SEPARATION = 0.0f;
	const float MAX_SEPARATION = 10.0f;
	const float MIN_ALIGNMENT = 0.0f;
	const float MAX_ALIGNMENT = 10.0f;
	const float MIN_COHESION = 0.0f;
	const float MAX_COHESION = 10.0f;
	const float MIN_MARGIN = 0.0f;
	const float MAX_MARGIN = 0.3f;
	const float MIN_SPEED = 1e-4f;
	const float MAX_SPEED = 0.8f;
	const float MIN_VISIBILITY = 0.04f;
	const float MAX_VISIBILITY = 0.5f;
	const float MIN_CELL_LEN = 2.0f * MIN_VISIBILITY;
	const float MIN_TURN = 1.0f;
	const float MAX_TURN = 10.0f;

	float separation = 1e-1f / 5.0f;
	float alignment = 3.0f;
	float cohesion = 5.0f;
	float margin = 0.1f;
	float speed = 0.4f;
	float visibility = 0.2f;
	float turn = 1.0f;*/

	const float MIN_SEPARATION = 0.0f;
	const float MAX_SEPARATION = 0.3f;
	const float SEPARATION_SCALING = MAX_SEPARATION / 10.0f;
	const float MIN_ALIGNMENT = 0.0f;
	const float MAX_ALIGNMENT = 10.0f;
	const float MIN_COHESION = 0.0f;
	const float MAX_COHESION = 10.0f;
	const float MIN_MARGIN = 0.0f;
	const float MAX_MARGIN = AQUARIUM_LEN / 5.0f;
	const float MIN_MIN_SPEED = 0.001f;
	const float MAX_MAX_SPEED = 0.05f; 
	const float MIN_VISIBILITY = 0.04f;
	const float MAX_VISIBILITY = 0.5f;
	const float MIN_CELL_LEN = 2.0f * MIN_VISIBILITY;
	const float MIN_TURN = 1.0f;
	const float MAX_TURN = 50.0f;

	float separation = (MAX_SEPARATION - MIN_SEPARATION) / 2.0f;
	float alignment = (MAX_ALIGNMENT - MIN_ALIGNMENT) / 2.0f;
	float cohesion = (MAX_COHESION - MIN_COHESION) / 2.0f;
	float separation_alter = (MAX_SEPARATION - MIN_SEPARATION) / 2.0f;
	float alignment_alter = (MAX_ALIGNMENT - MIN_ALIGNMENT) / 2.0f;
	float cohesion_alter = (MAX_COHESION - MIN_COHESION) / 2.0f;
	float margin = 0.0f;
	float max_speed = 0.2f;
	float min_speed = 0.005f;
	float visibility = 0.1f;
	float turn = (MIN_TURN + MAX_TURN) / 2.0f;
};

class Fishes
{
public:
	int numberOfFishes, numberOfBlocks;
	std::vector<float3> positions;
	std::vector<float3> velocities;
	fishesParams params;


	// accessible on device only below
	cudaSOA dev_soa;


	Fishes(CreateFishesInfo*);
	~Fishes();
	void update(float frameTime);
	void updateGPU(float frameTime);
	void updateCPU(float frameTime);
};
