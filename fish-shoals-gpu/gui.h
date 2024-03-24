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
	float *sep, *sep_alter, *coh, *coh_alter, *alg, *alg_alter, *min_speed, *max_speed, *margin, *turn, *visibility;

	void update();

	Gui(Fishes* fishes);
};

