#include "device_launch_parameters.h"
#include "config.cuh"
#include "app.h"


int main(int argc, char* argv[]) {
	// Default values
    int width = 640;
    int height = 480;
	int numberOfFishes = FISH_COUNT;

	// YOU HAVE TO SET number of fishes by changing FISH_COUNT constant

	//if (argc == 4) {
	//	// Three arguments provided, set number of fishes, width and height
	//	//numberOfFishes = atoi(argv[1]); // in thousands
	//	width = atoi(argv[1]);
	//	height = atoi(argv[2]);
	//}
    if (argc == 3) {
        // Two arguments provided, set width and height
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
	//else if (argc == 2) {
	//	// One argument provided, set number of fishes
	//	numberOfFishes = atoi(argv[1]);
	//}
    else if (argc != 1) {
        ERROR("Incorrect number of arguments");
    }

	validateCudaStatus(cudaDeviceReset());
	validateCudaStatus(cudaSetDevice(0));

	AppCreateInfo appInfo;
	appInfo.width = width;
	appInfo.height = height;
	appInfo.numberOfFishes = numberOfFishes;
	App* app = new App(&appInfo);

	returnCode nextAction = returnCode::CONTINUE;
	while (nextAction == returnCode::CONTINUE) {
		nextAction = app->mainLoop();
	}
	
	delete app;
	cudaDeviceReset();

	return 0;
}
