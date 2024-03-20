#pragma once
#include "config.cuh"
#include "cube.h"
#include "fishes.cuh"

struct CreateAquariumInfo {
	int numberOfFishes;
	glm::vec3 position;
	glm::vec3 size;
};

class Aquarium : public Cube
{
public:
	int numberOfFishes;
	Fishes* fishes;	

	Aquarium(CreateAquariumInfo* createInfo);
	~Aquarium();
	void update(float frameTime);
};

