#pragma once
#include "config.cuh"
#include "device_launch_parameters.h"

struct CreateFishesInfo {
	int numberOfFishes;
	glm::vec3 aquariumSize;
};

struct fishesParams {
	const float MIN_SEPARATION = 0.0f;
	const float MAX_SEPARATION = 8.0f;
	const float MIN_ALIGNMENT = 0.0f;
	const float MAX_ALIGNMENT = 8.0f;
	const float MIN_COHESION = 0.0f;
	const float MAX_COHESION = 8.0f;
	const float MIN_MARGIN = 0.0f;
	const float MAX_MARGIN = 0.3f;
	const float MIN_SPEED = 1e-4f;
	const float MAX_SPEED = 0.8f;
	const float MIN_VISIBILITY = 0.04f;
	const float MAX_VISIBILITY = 0.5f;

	float separation = 1e-1f / 5.0f;
	float alignment = 3.0f;
	float cohesion = 5.0f;
	float margin = 0.1f;
	float speed = 0.4f;
	float visibility = 0.2f;
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
	void update();
};
