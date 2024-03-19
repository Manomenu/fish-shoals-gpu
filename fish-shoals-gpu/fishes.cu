#include "fishes.cuh"

__global__ void updateGrid1Kernel(cudaSOA soa, float visibility)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	glm::vec3 fish_pos = soa.positions[tid];
	int grid_MAX_ID = (int)glm::ceil(AQUARIUM_LEN / visibility) - 1;
	
	int gridX_ID = glm::clamp((int)glm::floor((fish_pos.x - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);
	int gridY_ID = glm::clamp((int)glm::floor((fish_pos.y - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);
	int gridZ_ID = glm::clamp((int)glm::floor((fish_pos.z - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);

	soa.grid.cells[tid] = 
		gridZ_ID * grid_MAX_ID * grid_MAX_ID 
		+ gridY_ID * grid_MAX_ID 
		+ gridX_ID;
	soa.grid.fishesIDs[tid] = tid;
}

__global__ void updateGrid2Kernel(cudaSOA soa)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	int cell = soa.grid.cells[tid];

	if (tid == 0)
	{
		soa.grid.starts[cell] = tid;
		soa.grid.ends[soa.grid.cells[FISH_COUNT - 1]] = FISH_COUNT;
	}
	else
	{
		int previous_cell = soa.grid.cells[tid - 1];

		if (cell != previous_cell)
		{
			soa.grid.starts[cell] = tid;
			soa.grid.ends[previous_cell] = tid;
		}
	}
}

Fishes::Fishes(CreateFishesInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	numberOfBlocks = (numberOfFishes + MAX_THREADS + 1) / MAX_THREADS;

	positions = std::vector<float3>(numberOfFishes);
	velocities = std::vector<float3>(numberOfFishes);

	int grid_divided = (int)glm::ceil((float)AQUARIUM_LEN / (float)params.MIN_VISIBILITY / 2.0f);
	int grid_MAX_VALUE = grid_divided * grid_divided * grid_divided;

	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.cells, 
		sizeof(int) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.fishesIDs,
		sizeof(int) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.velocities,
		sizeof(float3) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.positions,
		sizeof(float3) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.starts,
		sizeof(int) * grid_MAX_VALUE
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.ends,
		sizeof(int) * grid_MAX_VALUE
	));

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

	validateCudaStatus(cudaMemcpy(
		dev_soa.velocities,
		velocities.data(),
		sizeof(float3) * numberOfFishes,
		cudaMemcpyHostToDevice
	));

	validateCudaStatus(cudaMemcpy(
		dev_soa.positions,
		positions.data(),
		sizeof(float3) * numberOfFishes,
		cudaMemcpyHostToDevice
	));
}

void Fishes::update()
{
	updateGrid1Kernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa, params.visibility);
	validateCudaStatus(cudaGetLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	thrust::sort_by_key(
		thrust::device, 
		dev_soa.grid.cells, 
		dev_soa.grid.cells + FISH_COUNT, 
		dev_soa.grid.fishesIDs
	);
	validateCudaStatus(cudaGetLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	updateGrid2Kernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa);
	validateCudaStatus(cudaGetLastError());
	validateCudaStatus(cudaDeviceSynchronize());
}

Fishes::~Fishes()
{
	validateCudaStatus(cudaFree(dev_soa.positions));
	validateCudaStatus(cudaFree(dev_soa.velocities));
	validateCudaStatus(cudaFree(dev_soa.grid.cells));
	validateCudaStatus(cudaFree(dev_soa.grid.ends));
	validateCudaStatus(cudaFree(dev_soa.grid.starts));
	validateCudaStatus(cudaFree(dev_soa.grid.fishesIDs));
}
