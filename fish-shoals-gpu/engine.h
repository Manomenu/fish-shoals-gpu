#pragma once
#include "config.cuh"
#include "scene.h"
#include "shader.h"
#include "rectangle_model.h"
#include "gui_model.h"
#include "material.h"

class Engine {
public:
	Engine(int width, int height);
	~Engine();

	void createMaterials();
	void createModels();
	void render(Scene* scene);

	Shader* shader;
	Material* woodMaterial;
	RectangleModel* cubeModel;
	GuiModel* guiModel;
	int width, height;
};

