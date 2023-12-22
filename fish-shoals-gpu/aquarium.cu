#include "aquarium.cuh"


Aquarium::Aquarium(CreateAquariumInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	position = createInfo->position;
	size = createInfo->size;
	fishPhysics = std::vector<FishPhysics>(numberOfFishes);
	fishTransformations = std::vector<FishTransformation>(numberOfFishes);

	setDefaultFishData();
}

void Aquarium::update()
{
	// use cuda kernel ;)
}

void Aquarium::setDefaultFishData()
{
	float minX = -size.x / 2.0f * 0.8f;
	float minY = -size.y / 2.0f * 0.8f;
	float minZ = -size.z / 2.0f * 0.8f;
	float maxX = size.x / 2.0f * 0.8f;
	float maxY = size.y / 2.0f * 0.8f;
	float maxZ = size.z / 2.0f * 0.8f;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> xDist(minX, maxX - 0.001f);
	std::uniform_real_distribution<float> yDist(minY, maxY - 0.002f);
	std::uniform_real_distribution<float> zDist(minZ, maxZ);
	std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

	std::for_each(
		std::execution::par,
		fishTransformations.begin(),
		fishTransformations.end(),
		[&](FishTransformation& fish)
		{
			float randx = xDist(gen);
			float randy = yDist(gen);
			float randz = zDist(gen);
			float dx = Dist(gen);
			float dy = Dist(gen);
			float dz = Dist(gen);

			fish.position = { randx, randy, randz };
			fish.direction = { dx, dy, dz };
		}
	);
	std::for_each(
		std::execution::par,
		fishPhysics.begin(),
		fishPhysics.end(),
		[&](FishPhysics& fish)
		{
			const auto& correspondingTransformation = fishTransformations[&fish - &fishPhysics[0]];
			fish.velocity = correspondingTransformation.direction;
		}
	);
}

