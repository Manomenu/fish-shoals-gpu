#include "aquarium.cuh"

Aquarium::Aquarium(CreateAquariumInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	position = createInfo->position;
	size = createInfo->size;

	CreateFishesInfo fishesInfo;
	fishesInfo.numberOfFishes = numberOfFishes;
	fishesInfo.aquariumSize = size;
	fishes = new Fishes(&fishesInfo);
}

Aquarium::~Aquarium()
{
}

void Aquarium::update(float frameTime)
{
	fishes->update(frameTime);
}

