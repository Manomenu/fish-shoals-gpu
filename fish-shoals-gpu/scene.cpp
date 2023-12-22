#include "scene.h"

Scene::Scene(int numberOfFishes) {

	PlayerCreateInfo playerInfo;
	playerInfo.eulers = { 0.0f, 90.0f,0.0f };
	playerInfo.position = { 0.0f, 0.0f, 1.0f };
	player = new Player(&playerInfo);

	CreateAquariumInfo aquariumInfo;
	aquariumInfo.position = { 0.0f, 0.0f, 0.0f };
	aquariumInfo.size = { AQUARIUM_SIZE };
	aquariumInfo.numberOfFishes = numberOfFishes;
	aquarium = new Aquarium(&aquariumInfo);

	gui = new Gui();
}

Scene::~Scene() {
	delete aquarium;
	delete player;
}

void Scene::update(float rate) {
	gui->update();
	player->update();
}

void Scene::movePlayer(glm::vec3 dPos) {
	player->position += dPos;
}

void Scene::spinPlayer(glm::vec3 dEulers) {
	player->eulers += dEulers;

	if (player->eulers.z < 0) {
		player->eulers.z += 360;
	}
	else if (player->eulers.z > 360) {
		player->eulers.z -= 360;
	}

	player->eulers.y = std::max(std::min(player->eulers.y, 179.0f), 1.0f);
}