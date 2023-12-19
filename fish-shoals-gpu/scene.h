#pragma once
#include "config.cuh"
#include "cube.h"
#include "player.h"
#include "gui.h"

class Scene {
public:
	Scene();
	~Scene();
	void update(float rate);
	void movePlayer(glm::vec3 dPos);
	void spinPlayer(glm::vec3 dEulers);

	Cube* cube;
	Player* player;
	Gui* gui;
};

