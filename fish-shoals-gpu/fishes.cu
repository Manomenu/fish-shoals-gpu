#include "fishes.cuh"

__device__ int getFishCell(glm::vec3 fish_pos, float cellLen)
{
	int gridIDsCount = (int)glm::ceil(AQUARIUM_LEN / cellLen);

	int gridX_ID = glm::clamp((int)glm::floor((fish_pos.x + AQUARIUM_LEN / 2.0f) / cellLen),
		0, gridIDsCount - 1);
	int gridY_ID = glm::clamp((int)glm::floor((fish_pos.y + AQUARIUM_LEN / 2.0f) / cellLen),
		0, gridIDsCount - 1);
	int gridZ_ID = glm::clamp((int)glm::floor((fish_pos.z + AQUARIUM_LEN / 2.0f) / cellLen),
		0, gridIDsCount - 1);

	return gridX_ID + gridY_ID * gridIDsCount + gridZ_ID * gridIDsCount * gridIDsCount;
}

__global__ void updateGrid1Kernel(cudaSOA soa, float visibility)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	soa.grid.cells[tid] = getFishCell(soa.positions[tid], 2 * visibility); // cell length = 2 * fish visibility :)
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

__device__ void checkFishNeighbourhood(
	cudaSOA& soa,
	fishesParams& fishesParams, 
	fishData& fishData, 
	int checkout_cell_ID, 
	int currentFishId
)
{
	int start = soa.grid.starts[checkout_cell_ID];
	if (start == -1) return;
	int end = soa.grid.ends[checkout_cell_ID];

	for (int i = start; i < end; ++i)
	{
		int fishId = soa.grid.fishesIDs[i];
		if (currentFishId == fishId) continue;

		glm::vec3 position_diff = soa.positions[currentFishId] - soa.positions[fishId];
		float dist = glm::length(position_diff);

		if (dist < fishesParams.visibility)
		{
			int mod2 = fishId % 2 == 0 ? -1 : 1;
			float dist2 = dist * dist;

			fishData.separation_factor += (dist2 > 1e-8)
				? glm::normalize(position_diff) / (dist2)
				: glm::vec3(
					 mod2 * 0.001f, 0, -mod2 * 0.001f
				);
			fishData.velocity += soa.velocities[fishId];
			fishData.position += soa.positions[fishId];
			++fishData.numberOfNeighbours;
		}
	}
}

__device__ glm::vec3 fishGroupBehaviourVelocityFactor(
	cudaSOA& soa, 
	fishesParams& fishesParams, 
	int tid
)
{
	fishData fishData;

	float cellLen = 2 * fishesParams.visibility;
	int grid_division = (int)glm::ceil(AQUARIUM_LEN / cellLen);
	int grid_division2 = grid_division * grid_division;
	int coords_count = grid_division * grid_division2;
	float coord_begin = -AQUARIUM_LEN / 2.0f;

	int cell = getFishCell(soa.positions[tid], cellLen);
	int X = cell % grid_division;
	int Y = (cell / grid_division) % grid_division;
	int Z = cell / (grid_division2);
	
	int xDist = (soa.positions[tid].x >= coord_begin + (X + 0.5) * cellLen)
		? 1 
		: -1;
	int yDist = (soa.positions[tid].y >= coord_begin + (Y + 0.5) * cellLen) 
		? grid_division 
		: -grid_division;
	int zDist = (soa.positions[tid].z >= coord_begin / 2.0f + (Z + 0.5) * cellLen) 
		? grid_division2 
		: -grid_division2;

	int checkout_cell_ID;
	for (int layer = 0; layer < 2; ++layer) // CHECK 8 cells
	{
		checkout_cell_ID = cell + layer * zDist;
		if (checkout_cell_ID >= 0 
			&& checkout_cell_ID < coords_count)
			checkFishNeighbourhood(soa, fishesParams, fishData, checkout_cell_ID, tid);

		checkout_cell_ID = cell + xDist + layer * zDist;
		if (checkout_cell_ID >= 0 
			&& checkout_cell_ID < coords_count
			&& checkout_cell_ID / grid_division == (checkout_cell_ID - xDist) / grid_division)
			checkFishNeighbourhood(soa, fishesParams, fishData, checkout_cell_ID, tid);

		checkout_cell_ID = cell + yDist + layer * zDist;
		if (checkout_cell_ID >= 0 
			&& checkout_cell_ID < coords_count
			&& checkout_cell_ID / grid_division2 == (checkout_cell_ID - yDist) / grid_division2)
			checkFishNeighbourhood(soa, fishesParams, fishData, checkout_cell_ID, tid);

		checkout_cell_ID = cell + xDist + yDist + layer * zDist;
		if (checkout_cell_ID >= 0 
			&& checkout_cell_ID < coords_count
			&& checkout_cell_ID / grid_division == (checkout_cell_ID - xDist) / grid_division
			&& checkout_cell_ID / grid_division2 == (checkout_cell_ID - yDist) / grid_division2) 
			checkFishNeighbourhood(soa, fishesParams, fishData, checkout_cell_ID, tid);
	}

	if (fishData.numberOfNeighbours == 0)
	{
		return glm::vec3(0);
	}

	fishData.velocity /= fishData.numberOfNeighbours;
	fishData.position /= fishData.numberOfNeighbours;
	fishData.alignment_factor = fishData.velocity - soa.velocities[tid];
	fishData.cohesion_factor = fishData.position - soa.positions[tid];

	return fishesParams.separation * fishesParams.SEPARATION_SCALING * fishData.separation_factor
		+ fishesParams.cohesion * fishData.cohesion_factor
		+ fishesParams.alignment * fishData.alignment_factor;
}

__device__ glm::vec3 wallVelocityFactor(glm::vec3 pos, fishesParams& fishesParams, glm::vec3 vel)
{
	glm::vec3 velocity_wall_factor = glm::vec3(0, 0, 0);
	float half_len = AQUARIUM_LEN / 2.0f;
	float dist = glm::length(vel);

	if (half_len - pos.x < fishesParams.margin)
		velocity_wall_factor.x -= fishesParams.turn * dist * (pos.x + -half_len + fishesParams.margin);
	if (pos.x + half_len < fishesParams.margin)
		velocity_wall_factor.x += fishesParams.turn * dist * (-half_len - pos.x + fishesParams.margin);
	if (half_len - pos.y < fishesParams.margin)
		velocity_wall_factor.y -= fishesParams.turn * dist * (pos.y - half_len + fishesParams.margin);
	if (pos.y + half_len < fishesParams.margin)
		velocity_wall_factor.y += fishesParams.turn * dist * (-half_len - pos.y + fishesParams.margin);
	if (half_len - pos.z < fishesParams.margin)
		velocity_wall_factor.z -= fishesParams.turn * dist * (pos.z - half_len + fishesParams.margin);
	if (pos.z + half_len < fishesParams.margin)
		velocity_wall_factor.z += fishesParams.turn * dist * (-half_len - pos.z + fishesParams.margin);

	return velocity_wall_factor;
}

__device__ glm::vec3 speedLimit(fishesParams& fishesParams, glm::vec3 vel)
{
	float len = glm::length(vel);

	if (len < fishesParams.min_speed)
		return fishesParams.min_speed * glm::normalize(vel);
	if (len > fishesParams.max_speed)
		return fishesParams.max_speed * glm::normalize(vel);
	return vel;
}

__global__ void updateSOAKernel(cudaSOA soa, fishesParams params, float frameTime)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= FISH_COUNT)
		return;

	glm::vec3 new_velocity(0);
	new_velocity += soa.velocities[tid];
	new_velocity += fishGroupBehaviourVelocityFactor(soa, params, tid) * frameTime;
	new_velocity += wallVelocityFactor(soa.positions[tid], params, new_velocity) * frameTime;

	new_velocity = speedLimit(params, new_velocity);

	// this just just to not override old value whilst other threads still work {
	soa.velocities_P[tid] = new_velocity; 
	soa.positions_P[tid] = soa.positions[tid] + frameTime * new_velocity;

	__syncthreads();

	soa.positions[tid] = soa.positions_P[tid];
	soa.velocities[tid] = soa.velocities_P[tid];
	// }
}

Fishes::Fishes(CreateFishesInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	numberOfBlocks = (numberOfFishes + MAX_THREADS - 1) / MAX_THREADS;

	positions = std::vector<float3>(numberOfFishes);
	velocities = std::vector<float3>(numberOfFishes);

	int grid_divided = (int)glm::ceil(AQUARIUM_LEN / params.MIN_CELL_LEN);
	int gridIDsCount = grid_divided * grid_divided * grid_divided;

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
		(void**)&dev_soa.velocities_P,
		sizeof(float3) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.positions_P,
		sizeof(float3) * numberOfFishes
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.starts,
		sizeof(int) * gridIDsCount
	));
	validateCudaStatus(cudaMalloc(
		(void**)&dev_soa.grid.ends,
		sizeof(int) * gridIDsCount
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
	frameTime /= 16.0f;

	int grid_divided = (int)glm::ceil(AQUARIUM_LEN / params.MIN_CELL_LEN);
	int grid_MAX_VALUE = grid_divided * grid_divided * grid_divided;

	validateCudaStatus(cudaMemset(dev_soa.grid.cells, 0, sizeof(int) * FISH_COUNT));
	validateCudaStatus(cudaMemset(dev_soa.grid.fishesIDs, 0, sizeof(int) * FISH_COUNT));
	validateCudaStatus(cudaMemset(dev_soa.grid.starts, -1, sizeof(int) * grid_MAX_VALUE));
	validateCudaStatus(cudaMemset(dev_soa.grid.ends, -1, sizeof(int) * grid_MAX_VALUE));

	updateGrid1Kernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa, params.visibility);
	validateCudaStatus(cudaPeekAtLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	thrust::sort_by_key(
		thrust::device, 
		dev_soa.grid.cells, 
		dev_soa.grid.cells + FISH_COUNT, 
		dev_soa.grid.fishesIDs
	);
	validateCudaStatus(cudaPeekAtLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	updateGrid2Kernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa);
	validateCudaStatus(cudaPeekAtLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	updateSOAKernel<<<numberOfBlocks, MAX_THREADS>>>(dev_soa, params, frameTime);
	validateCudaStatus(cudaPeekAtLastError());
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
	validateCudaStatus(cudaFree(dev_soa.positions_P));
	validateCudaStatus(cudaFree(dev_soa.velocities_P));
}
