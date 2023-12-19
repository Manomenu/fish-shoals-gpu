#pragma once
#include "config.cuh"
#include "scene.h"
#include "engine.h"

struct AppCreateInfo {
	int width;
	int height;
};

enum class returnCode {
	CONTINUE, QUIT
};

class App {
public:
	App(AppCreateInfo* createInfo);
	returnCode mainLoop();
	~App();
private:
	GLFWwindow* makeWindow();
	returnCode processInput();
	void calculateFrameRate();
	void setUpImgui();
	void updateWindowSize();

	GLFWwindow* window;
	int width, height;
	Scene* scene;
	Engine* renderer;

	double lastTime, currentTime;
	int numFrames;
	float frameTime;

#pragma region Callbacks
	static void framebufferSizeCallback(GLFWwindow*, int, int);

	static bool windowSizeChanged;
#pragma endregion
};

