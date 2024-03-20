#pragma once
#include "config.cuh"
#include "fishes.cuh"

typedef struct fishesParams simulationParams;

class Gui
{
public:
	bool show_window = true;
	float fps = 0;
	simulationParams simulationParams;

	void update();

	Gui(Fishes* fishes);
};

