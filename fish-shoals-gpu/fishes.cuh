#pragma once
#include "config.cuh"

struct CreateFishesInfo {
	int numberOfFishes;
	glm::vec3 aquariumSize;
};

struct fishesParams {
	const float MIN_SEPARATION = 0;
	const float MAX_SEPARATION = 8;
	const float MIN_ALIGNMENT = 0;
	const float MAX_ALIGNMENT = 8;
	const float MIN_COHESION = 0;
	const float MAX_COHESION = 8;
	const float MIN_MARGIN = 0;
	const float MAX_MARGIN = 0.3f;
	const float MIN_SPEED = 1e-4;
	const float MAX_SPEED = 0.8f;
	const float MIN_VISIBILITY = 0.04f;
	const float MAX_VISIBILITY = 0.5f;

	float separation = 1e-1 / 5;
	float alignment = 3;
	float cohesion = 5;
	float margin = 0.1f;
	float speed = 0.4f;
	float visibility = 0.2f;
};

class Fishes
{
public:
	int numberOfFishes;
	std::vector<float3> positions;
	std::vector<float3> velocities;
	fishesParams params;


	// accessible on device only below
	cudaSOA cudaSOA;


	Fishes(CreateFishesInfo*);
	~Fishes();
	void update();
};
