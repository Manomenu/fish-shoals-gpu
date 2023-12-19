#pragma once
#include "config.cuh"

struct PlayerCreateInfo {
	glm::vec3 position, eulers;
};

class Player {
public:
	glm::vec3 position, eulers, up, forwards, right;

	Player(PlayerCreateInfo* createInfo);
	void update();
};

