#include "engine.h"

Engine::Engine(int width, int height, int numberOfFishes) {

	shader = new Shader("vertex.txt", "fragment.txt");
	shader->use();

	shader->setInt("basicTexture", 0);

	this->width = width;
	this->height = height;
	this->numberOfFishes = numberOfFishes;
	fishModels = std::vector<PyramidModel*>(numberOfFishes);

	float aspectRatio = (float)width / float(height);
	//set up framebuffer
	glClearColor(BACKGROUND_COLOR);
	glEnable(GL_DEPTH_TEST);
	glm::mat4 projection_transform = glm::perspective(FOVY, aspectRatio, NEAR, FAR);
	shader->setMat4("projection", projection_transform);

	createModels();
	createMaterials();
}

Engine::~Engine() {
	delete aquariumModel;
	delete shader;
}

void Engine::createModels() {
	RectangleModelCreateInfo aquariumInfo;
	aquariumInfo.size = { AQUARIUM_SIZE };
	aquariumModel = new RectangleModel(&aquariumInfo);

	guiModel = new GuiModel();

	FishesModelCreateInfo fishesInfo;
	fishesInfo.numberOfFishes = numberOfFishes;
	fishesModel = new FishesModel(&fishesInfo);
}

void Engine::createMaterials() {
	// i can create wood for example
}

void Engine::render(Scene* scene)
{
	// prepraring universal transforms
	glm::mat4 view_transform{
		glm::lookAt(
			scene->player->position,
			scene->player->position + scene->player->forwards,
			scene->player->up
		)
	};
	shader->setMat4("view", view_transform);

	float aspectRatio = (float)width / float(height);
	glm::mat4 projection_transform = glm::perspective(FOVY, aspectRatio, NEAR, FAR);
	shader->setMat4("projection", projection_transform);

	// render scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	aquariumModel->render(scene->aquarium, nullptr, shader, false);
	
	fishesModel->render(shader, scene->aquarium->fishTransformations.data());
	

	// render imgui
	guiModel->render();
}
