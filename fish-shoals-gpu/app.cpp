#include "app.h"

bool App::windowSizeChanged = false;

App::App(AppCreateInfo* createInfo) {
	this->width = createInfo->width;
	this->height = createInfo->height;
	this->numberOfFishes = createInfo->numberOfFishes;

	lastTime = glfwGetTime();
	numFrames = 0;
	frameTime = 16.0f;

	window = makeWindow();
	
	setUpImgui();

	renderer = new Engine(width, height, numberOfFishes);
	scene = new Scene(numberOfFishes);
}

void App::setUpImgui()
{
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 450");
}

GLFWwindow* App::makeWindow() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Fish shoal", NULL, NULL);
	
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "GLAD initialization failed\n";
		return NULL;
	}

	glViewport(0, 0, width, height);
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

	return window;
}

void App::framebufferSizeCallback(GLFWwindow * window, int width, int height)
{
	glViewport(0, 0, width, height);
	windowSizeChanged = true;
}

returnCode App::processInput() {

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		scene->camera->ProcessKeyboard(Camera_Movement::FORWARD, frameTime);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		scene->camera->ProcessKeyboard(Camera_Movement::LEFT, frameTime);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		scene->camera->ProcessKeyboard(Camera_Movement::BACKWARD, frameTime);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		scene->camera->ProcessKeyboard(Camera_Movement::RIGHT, frameTime);
	}

	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
	{
		lock = true;
	}
	if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
	{
		lock = false;
	}

	if (lock)
	{
		ImGui::SetMouseCursor(ImGuiMouseCursor_None);

		double mouse_x, mouse_y;
		glfwGetCursorPos(window, &mouse_x, &mouse_y);
		glfwSetCursorPos(window, static_cast<double>(width / 2), static_cast<double>(height / 2));

		float delta_x{ static_cast<float>(mouse_x - static_cast<double>(width / 2)) };
		float delta_y{ static_cast<float>(mouse_y - static_cast<double>(height / 2)) };

		scene->camera->ProcessMouseMovement(delta_x * 2, -delta_y * 2);
	}
	else
	{
		ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		return returnCode::QUIT;
	}
	return returnCode::CONTINUE;
}

returnCode App::mainLoop() {

	returnCode nextAction{ processInput() };
	glfwPollEvents();

	//update
	calculateFrameRate();
	updateWindowSize();

	if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
	{
		pause = true;
	}
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
	{
		pause = false;
	}
	if (!pause)
	{
		scene->update(frameTime);
	}

	//draw
	renderer->render(scene);
	glfwSwapBuffers(window);

	

	return nextAction;
}

void App::updateWindowSize()
{
	if (windowSizeChanged)
	{
		glfwGetWindowSize(window, &width, &height);
		renderer->width = width;
		renderer->height = height;

		windowSizeChanged = false;
	}
}

App::~App() {
	//free memory
	delete scene;
	delete renderer;
	glfwDestroyWindow(window);
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
}
#define NOMINMAX
#include<windows.h> 
void App::calculateFrameRate() {
	currentTime = glfwGetTime();
	double delta = currentTime - lastTime;

	if (delta >= 1) {
		int framerate{ std::max(1, int(numFrames / delta)) };
		std::stringstream title;
		title << "Fish shoals simulation (running at " << framerate << " fps)";
		glfwSetWindowTitle(window, title.str().c_str());
		lastTime = currentTime;
		numFrames = -1;
		frameTime = std::min(float(delta), 1.0f / 60.0f);
		Sleep(1000.0f / 60.0f - frameTime);
	}

	++numFrames;
}