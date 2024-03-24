#include "engine.cuh"

Engine::Engine(int width, int height, int numberOfFishes) {

	this->width = width;
	this->height = height;
	this->numberOfFishes = numberOfFishes;

	float aspectRatio = (float)width / float(height);
	glm::mat4 projection_transform = glm::perspective(FOVY, aspectRatio, NEAR, FAR);

	shader = new Shader("vertex.txt", "fragment.txt");
	shader->use();
	shader->setInt("basicTexture", 0);
	shader->setMat4("projection", projection_transform);

	fishesShader = new Shader("fishesVertex.txt", "fishesFragment.txt");
	fishesShader->use();
	fishesShader->setMat4("projection", projection_transform);

	glClearColor(BACKGROUND_COLOR);
	glEnable(GL_DEPTH_TEST);

	createModels();
}

Engine::~Engine() {
	delete aquariumModel;
	delete shader;
	delete fishesShader;
	delete guiModel;
	delete fishesModel;
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

void Engine::render(Scene* scene)
{
	// prepraring universal transforms
	const glm::mat4 view_transform{
		scene->camera->GetViewMatrix()
	};

	float aspectRatio = (float)width / float(height);
	const glm::mat4 projection_transform = glm::perspective(FOVY, aspectRatio, NEAR, FAR);

	// render scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	aquariumModel->render(scene->aquarium, shader, false, view_transform, projection_transform);

	fishesModel->render(fishesShader, scene->aquarium->fishes, view_transform, projection_transform);

	// render imgui
	guiModel->render();
}
