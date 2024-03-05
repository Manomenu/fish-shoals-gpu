#pragma once
#include "config.cuh"
#include "cube.h"
#include "fish_transformation.h"
#include "fish_physics.h"
#include "device_launch_parameters.h"

struct CreateAquariumInfo {
	int numberOfFishes;
	glm::vec3 position;
	glm::vec3 size;
};

class Aquarium : public Cube
{
public:
	int numberOfFishes;
	std::vector<float> fishTransformations;
	std::vector<float> fishPhysics;

	// accessible on device only below
	float* dev_fishTransformations;
	float* dev_fishPhysics;

	Aquarium(CreateAquariumInfo* createInfo);
	~Aquarium();
	void update();
	void setDefaultFishData();
};

