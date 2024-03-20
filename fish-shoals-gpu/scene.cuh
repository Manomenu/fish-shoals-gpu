#pragma once
#include "config.cuh"
#include "player.h"
#include "gui.h"
#include "aquarium.cuh"

class Scene {
public:
	Scene(int numberOfFishes);
	~Scene();
	void update(float frameTime);
	void movePlayer(glm::vec3 dPos);
	void spinPlayer(glm::vec3 dEulers);

	Aquarium* aquarium;
	Player* player;
	Gui* gui;
};

