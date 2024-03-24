#include "scene.cuh"

Scene::Scene(int numberOfFishes) {

	camera = new Camera();

	CreateAquariumInfo aquariumInfo;
	aquariumInfo.position = { 0.0f, 0.0f, 0.0f };
	aquariumInfo.size = { AQUARIUM_SIZE };
	aquariumInfo.numberOfFishes = numberOfFishes;
	aquarium = new Aquarium(&aquariumInfo);

	gui = new Gui(aquarium->fishes);
}

Scene::~Scene() {
	delete aquarium;
	delete camera;
}

void Scene::update(float frameTime) {
	gui->update();
	aquarium->update(frameTime);
}
