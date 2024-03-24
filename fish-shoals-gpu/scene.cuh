#pragma once
#include "config.cuh"
#include "gui.h"
#include "aquarium.cuh"
#include "camera.h"

class Scene {
public:
	Scene(int numberOfFishes);
	~Scene();
	void update(float frameTime);

	Aquarium* aquarium;
	Camera* camera;
	Gui* gui;
};

