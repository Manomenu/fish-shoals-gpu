#pragma once
#include "config.cuh"
#include "scene.cuh"
#include "shader.h"
#include "rectangle_model.h"
#include "gui_model.h"
#include "fishes_model.cuh"

class Engine {
public:
	Engine(int width, int height, int numberOfFishes);
	~Engine();

	void createModels();
	void render(Scene* scene);

	Shader* shader, * fishesShader;
	RectangleModel* aquariumModel;
	GuiModel* guiModel;
	FishesModel* fishesModel;
	int width, height, numberOfFishes;
};

