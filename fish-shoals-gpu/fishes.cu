#include "fishes.cuh"

Fishes::Fishes(CreateFishesInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;

	positions = std::vector<float3>(numberOfFishes);
	velocities = std::vector<float3>(numberOfFishes);

	float minX = -createInfo->aquariumSize.x / 2.0f * 0.8f;
	float minY = -createInfo->aquariumSize.y / 2.0f * 0.8f;
	float minZ = -createInfo->aquariumSize.z / 2.0f * 0.8f;
	float maxX = createInfo->aquariumSize.x / 2.0f * 0.8f;
	float maxY = createInfo->aquariumSize.y / 2.0f * 0.8f;
	float maxZ = createInfo->aquariumSize.z / 2.0f * 0.8f;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> xDist(minX, maxX - 0.001f);
	std::uniform_real_distribution<float> yDist(minY, maxY - 0.002f);
	std::uniform_real_distribution<float> zDist(minZ, maxZ);
	std::uniform_real_distribution<float> Dist(-1.0f, 1.0f);

	for (int i = 0; i < FISH_COUNT; ++i)
	{
		float3& fish = positions[i];

		float randx = xDist(gen);
		fish.x = randx;
		float randy = yDist(gen);
		fish.y = randy;
		float randz = zDist(gen);
		fish.z = randz;
	}
	//std::for_each(
	//	std::execution::par,
	//	positions.begin(),
	//	positions.end(),
	//	[&](float3& fish)
	//	{
	//		
	//	}
	//);

	for (int i = 0; i < FISH_COUNT; ++i)
	{
		float3& fish = velocities[i];

		float dx = Dist(gen);
		fish.x = dx;
		float dy = Dist(gen);
		fish.y = dy;
		float dz = Dist(gen);
		fish.y = dz;
	}

	validateCudaStatus(cudaMalloc(
		(void**)&cudaSOA.dev_velocities,
		sizeof(float3) * numberOfFishes
	));

	validateCudaStatus(cudaMemcpy(
		cudaSOA.dev_velocities,
		velocities.data(),
		sizeof(float3) * numberOfFishes,
		cudaMemcpyHostToDevice
	));

	validateCudaStatus(cudaMalloc(
		(void**)&cudaSOA.dev_positions,
		sizeof(float3) * numberOfFishes
	));

	validateCudaStatus(cudaMemcpy(
		cudaSOA.dev_positions,
		positions.data(),
		sizeof(float3) * numberOfFishes,
		cudaMemcpyHostToDevice
	));
}

void Fishes::update()
{
	int xd = 3;
}

Fishes::~Fishes()
{
	validateCudaStatus(cudaFree(cudaSOA.dev_positions));
	validateCudaStatus(cudaFree(cudaSOA.dev_velocities));
}
