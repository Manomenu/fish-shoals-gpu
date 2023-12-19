#include "engine.h"

Engine::Engine(int width, int height) {

	shader = new Shader("vertex.txt", "fragment.txt");
	shader->use();

	shader->setInt("basicTexture", 0);

	this->width = width;
	this->height = height;

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
	delete woodMaterial;
	delete cubeModel;
	delete shader;
}

void Engine::createModels() {
	RectangleModelCreateInfo cubeInfo;
	cubeInfo.size = { 2.0f, 1.0f, 1.0f };
	cubeModel = new RectangleModel(&cubeInfo);

	guiModel = new GuiModel();
}

void Engine::createMaterials() {
	MaterialCreateInfo materialInfo;
	materialInfo.filename = "wood.jpeg";
	woodMaterial = new Material(&materialInfo);
}

void Engine::render(Scene* scene) {

	//prepare shaders
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

	glm::mat4 model_transform{ glm::mat4(1.0f) };
	model_transform = glm::translate(model_transform, scene->cube->position);
	model_transform =
		model_transform * glm::eulerAngleXYZ(
			scene->cube->eulers.x, scene->cube->eulers.y, scene->cube->eulers.z
		);
	shader->setMat4("model", model_transform);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	shader->use();
	woodMaterial->use();
	glBindVertexArray(cubeModel->VAO);
	glDrawArrays(GL_TRIANGLES, 0, cubeModel->vertexCount);

	// render imgui
	guiModel->render();
}
