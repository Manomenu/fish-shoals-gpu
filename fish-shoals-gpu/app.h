#pragma once
#include "config.cuh"
#include "engine.cuh"

struct AppCreateInfo {
	int width;
	int height;
	int numberOfFishes;
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
	int numberOfFishes;
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

