#include "fishes.cuh"

__device__ int getFishCell(glm::vec3 fish_pos, float visibility)
{
	int grid_MAX_ID = (int)glm::ceil(AQUARIUM_LEN / visibility) - 1;

	int gridX_ID = glm::clamp((int)glm::floor((fish_pos.x - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);
	int gridY_ID = glm::clamp((int)glm::floor((fish_pos.y - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);
	int gridZ_ID = glm::clamp((int)glm::floor((fish_pos.z - AQUARIUM_LEN / 2.0f) / visibility),
		0, grid_MAX_ID);

	return gridX_ID + gridY_ID * grid_MAX_ID + gridZ_ID * grid_MAX_ID * grid_MAX_ID;
}

__global__ void updateGrid1Kernel(cudaSOA soa, float visibility)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	soa.grid.cells[tid] = getFishCell(soa.positions[tid], visibility);;	
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

__device__ glm::vec3 fishGroupBehaviourVelocityFactor(
	cudaSOA& soa, 
	fishesParams& fishesParams, 
	int tid
)
{
	glm::vec3 separation_factor(0);
	glm::vec3 velocity(0);
	glm::vec3 position(0);
	int numberOfNeighbours = 0;

	float R = 2 * fishesParams.visibility;
	int grid_division = (int)glm::ceil(AQUARIUM_LEN / R);
	int cell = getFishCell(soa.positions[tid], R);
	int X = cell % grid_division;
	int Y = (cell / grid_division) % grid_division;
	int Z = cell / (grid_division * grid_division);
	int coords_count = grid_division * grid_division * grid_division;
	int x_change = (soa.positions[tid].x >= -AQUARIUM_LEN / 2.0f + (X + 0.5) * R)
		? 1 
		: -1;
	int y_change = (soa.positions[tid].y >= -AQUARIUM_LEN / 2.0f + (Y + 0.5) * R) 
		? grid_division 
		: -grid_division;
	int z_change = (soa.positions[tid].z >= -AQUARIUM_LEN / 2.0f + (Z + 0.5) * R) 
		? grid_division * grid_division 
		: -grid_division * grid_division;
}

__global__ void updateSOAKernel(cudaSOA &dev_soa, fishesParams& params, float frameTime)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	soa.positions_bb[i] = glm::vec3(0);
	soa.velocities_bb[i] = glm::vec3(0);

	glm::vec3 new_vel;
	new_vel = soa.velocities[i] + fishGroupBehaviourVelocityFactor(soa, params, i) * frameTime;
	new_vel = turn_from_wall(soa.positions[i], params, new_vel) * frameTime;
	new_vel = speedLimitFactor(params, new_vel);

	glm::vec3 new_pos = soa.positions[i] + (float)d * new_vel;

	soa.velocities_bb[i] = new_vel;
	soa.positions_bb[i] = new_pos;
}

Fishes::Fishes(CreateFishesInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	numberOfBlocks = (numberOfFishes + MAX_THREADS + 1) / MAX_THREADS;

	positions = std::vector<float3>(numberOfFishes);
	velocities = std::vector<float3>(numberOfFishes);

	int grid_divided = (int)glm::ceil((float)AQUARIUM_LEN / (float)params.MIN_VISIBILITY);
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

void Fishes::update(float frameTime)
{
	int grid_divided = (int)glm::ceil((float)AQUARIUM_LEN / (float)params.MIN_VISIBILITY);
	int grid_MAX_VALUE = grid_divided * grid_divided * grid_divided;

	validateCudaStatus(cudaMemset(dev_soa.grid.cells, 0, sizeof(int) * FISH_COUNT));
	validateCudaStatus(cudaMemset(dev_soa.grid.fishesIDs, 0, sizeof(int) * FISH_COUNT));
	validateCudaStatus(cudaMemset(dev_soa.grid.starts, -1, sizeof(int) * grid_MAX_VALUE));
	validateCudaStatus(cudaMemset(dev_soa.grid.ends, -1, sizeof(int) * grid_MAX_VALUE));


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

	updateSOAKernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa, params, frameTime);
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
