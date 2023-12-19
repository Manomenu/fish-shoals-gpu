#include "device_launch_parameters.h"
#include "config.cuh"
#include "app.h"


int main(int argc, char* argv[]) {
    int width = 640;
    int height = 480;

    if (argc == 3) {
        // Two arguments provided, set width and height
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    else if (argc != 1) {
        ERROR("Incorrect number of arguments");
    }

	AppCreateInfo appInfo;
	appInfo.width = width;
	appInfo.height = height;
	App* app = new App(&appInfo);

	returnCode nextAction = returnCode::CONTINUE;
	while (nextAction == returnCode::CONTINUE) {
		nextAction = app->mainLoop();
	}

	delete app;

	return 0;
}
