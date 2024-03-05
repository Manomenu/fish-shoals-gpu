#include "aquarium.cuh"

Aquarium::Aquarium(CreateAquariumInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	position = createInfo->position;
	size = createInfo->size;
	fishPhysics = std::vector<float>(numberOfFishes * 3);
	fishTransformations = std::vector<float>(numberOfFishes * 3);

	setDefaultFishData();
}

Aquarium::~Aquarium()
{
	cudaFree(dev_fishPhysics);
	cudaFree(dev_fishTransformations);
}

void Aquarium::update()
{
	// use cuda kernel
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
		[&](float& fish)
		{
			float randx = xDist(gen);
			fish = randx;
		}
	);

	std::for_each(
		std::execution::par,
		fishPhysics.begin(),
		fishPhysics.end(),
		[&](float& fish)
		{
			float dx = Dist(gen);
			fish = dx;
		}
	);

	validateCudaStatus(cudaMalloc(
		(void**)&dev_fishPhysics, 
		sizeof(float) * numberOfFishes * 3
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_fishTransformations,
		sizeof(float) * numberOfFishes * 3
	));

	validateCudaStatus(cudaMemcpy(
		dev_fishPhysics,
		fishPhysics.data(),
		sizeof(float) * numberOfFishes * 3,
		cudaMemcpyHostToDevice
	));
	validateCudaStatus(cudaMemcpy(
		dev_fishTransformations,
		fishTransformations.data(),
		sizeof(float) * numberOfFishes * 3,
		cudaMemcpyHostToDevice
	));
}

