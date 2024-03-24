#pragma once
#include "config.cuh"
#include "fishes.cuh"

typedef fishesParams simulationParams;

class Gui
{
public:
	bool show_window = true;
	float fps = 0;
	simulationParams simulationParams;
	float *sep, *coh, *alg, *min_speed, *max_speed, *margin, *turn, *visibility;

	void update();

	Gui(Fishes* fishes);
};

