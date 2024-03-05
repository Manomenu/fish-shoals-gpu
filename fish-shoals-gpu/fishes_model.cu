#include "fishes_model.cuh"

FishesModel::FishesModel(FishesModelCreateInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	vertexCount = createInfo->numberOfFishes * 5 * 3;
	vertices = std::vector<float>(vertexCount);
	numberOfBlocks = numberOfFishes / 1000;

	std::vector<int> indicesSchema = std::vector<int>
	{
		0, 1, 3,
		1, 2, 3,
		0, 1, 4,
		1, 2, 4,
		2, 3, 4,
		3 ,0 ,4
	};

	std::vector<int> indices = std::vector<int>();

	for (int i = 0; i < numberOfFishes; ++i)
	{
		for (int j = 0; j < indicesSchema.size(); ++j)
		{
			indices.push_back(indicesSchema[j] + 5 * i);
		}
	}

	//glCreateBuffers(1, &VBO);
	//glCreateVertexArrays(1, &VAO);
	//glCreateBuffers(1, &EBO);

	////glBindVertexArray(VAO);
	////glBindBuffer(GL_ARRAY_BUFFER, VBO);
	////glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), 0, GL_DYNAMIC_DRAW);
	////glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	////glEnableVertexAttribArray(0);
	//glVertexArrayVertexBuffer(VAO, 0, VBO, 0, sizeof(float) * 3);
	//glNamedBufferStorage(VBO, vertexCount * sizeof(float), indices.data(), GL_DYNAMIC_STORAGE_BIT);
	//// 
	////glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	////glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int), indices.data(), GL_STATIC_DRAW);

	//glVertexArrayElementBuffer(VAO, EBO);
	//glNamedBufferStorage(EBO, indices.size() * sizeof(int), indices.data(), GL_STATIC_DRAW);
	//glEnableVertexArrayAttrib(VAO, 0);
	//glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
	//glVertexArrayAttribBinding(VAO, 0, 0);

	//validateCudaStatus(cudaMalloc((void**)&resource_vbo, sizeof(float) * vertexCount));

	//validateCudaStatus(cudaGraphicsGLRegisterBuffer(&resource_vbo, VBO, 
	//	cudaGraphicsRegisterFlagsWriteDiscard));
	//validateCudaStatus(cudaGraphicsMapResources(1, &resource_vbo));
	//validateCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_vbo_data, &vbo_size, resource_vbo));

	//validateCudaStatus(cudaMalloc((void**)&dev_fish_model, sizeof(float) * 15));
	//validateCudaStatus(cudaMemcpy(dev_fish_model, fishModel, sizeof(float) * 15, cudaMemcpyHostToDevice));


	glCreateBuffers(1, &VBO);
	glCreateBuffers(1, &EBO);
	glCreateVertexArrays(1, &VAO);
	
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int), indices.data(), GL_STATIC_DRAW);

	validateCudaStatus(cudaGraphicsGLRegisterBuffer(&resource_vbo, VBO, 0));
	validateCudaStatus(cudaGraphicsMapResources(1, &resource_vbo));
	validateCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_vbo_data, &vbo_size, resource_vbo));

	validateCudaStatus(cudaMalloc((void**)&dev_fishes_positions, sizeof(float) * numberOfFishes * 3));
}

FishesModel::~FishesModel()
{
	glDeleteBuffers(1, &EBO);
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
	validateCudaStatus(cudaGraphicsUnmapResources(1, &resource_vbo));
}

__global__ void setFishesVertices(float3* dev_vbo_data, float* fishPositions, float* fishDirections, 
	float* fishModel)
{
	float xd = fishPositions[0];
	fishPositions[0] = -33;
	xd = fishPositions[0];
	float xdd = fishDirections[0];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	dev_vbo_data[0].x = 2;
	
	int size = sizeof(dev_vbo_data) / sizeof(float3);
	int size2 = sizeof(fishPositions) / sizeof(float);
		/*dev_vbo_data[15 * tid + 1] = 2;
		dev_vbo_data[15 * tid + 2] = 2;

		dev_vbo_data[15 * tid + 3] = 2;
		dev_vbo_data[15 * tid + 4] = 2;
		dev_vbo_data[15 * tid + 5] = 2;

		dev_vbo_data[15 * tid + 6] = 2;
		dev_vbo_data[15 * tid + 7] = 2;
		dev_vbo_data[15 * tid + 8] = 2;

		dev_vbo_data[15 * tid + 9] = 2;
		dev_vbo_data[15 * tid + 10] = 2;
		dev_vbo_data[15 * tid + 11] = 2;

		dev_vbo_data[15 * tid + 12] = 2;
		dev_vbo_data[15 * tid + 13] = 2;
		dev_vbo_data[15 * tid + 14] = 2;*/

	//glm::mat4 transform = glm::mat4(1.0);
	//transform = glm::translate(transform, 
	//	{ 
	//		fishPositions[3 * tid + 0],
	//		fishPositions[3 * tid + 1],
	//		fishPositions[3 * tid + 2]
	//	});

	//glm::vec3 pos = transform * glm::vec4(
	//	*(fishModel->vertices + 0), 
	//	*(fishModel->vertices + 1),
	//	*(fishModel->vertices + 2),
	//	1
	//);

	//dev_vbo_data[15 * tid + 0] = pos.x;
	//dev_vbo_data[15 * tid + 1] = pos.y;
	//dev_vbo_data[15 * tid + 2] = pos.z;

	//pos = transform * glm::vec4(
	//	*(fishModel->vertices + 3),
	//	*(fishModel->vertices + 4),
	//	*(fishModel->vertices + 5),
	//	1
	//);

	//dev_vbo_data[15 * tid + 3] = pos.x;
	//dev_vbo_data[15 * tid + 4] = pos.y;
	//dev_vbo_data[15 * tid + 5] = pos.z;

	//pos = transform * glm::vec4(
	//	*(fishModel->vertices + 6),
	//	*(fishModel->vertices + 7),
	//	*(fishModel->vertices + 8),
	//	1
	//);

	//dev_vbo_data[15 * tid + 6] = pos.x;
	//dev_vbo_data[15 * tid + 7] = pos.y;
	//dev_vbo_data[15 * tid + 8] = pos.z;
	//
	//pos = transform * glm::vec4(
	//	*(fishModel->vertices + 9),
	//	*(fishModel->vertices + 10),
	//	*(fishModel->vertices + 11),
	//	1
	//);

	//dev_vbo_data[15 * tid + 9] = pos.x;
	//dev_vbo_data[15 * tid + 10] = pos.y;
	//dev_vbo_data[15 * tid + 11] = pos.z;

	//pos = transform * glm::vec4(
	//	*(fishModel->vertices + 12),
	//	*(fishModel->vertices + 13),
	//	*(fishModel->vertices + 14),
	//	1
	//);

	//dev_vbo_data[15 * tid + 12] = pos.x;
	//dev_vbo_data[15 * tid + 13] = pos.y;
	//dev_vbo_data[15 * tid + 14] = pos.z;



	/*dev_vbo_data[15 * tid + 0] = *(fishModel->vertices + 0) + fishPositions[tid].position.x;
	dev_vbo_data[15 * tid + 1] = *(fishModel->vertices + 1) + fishPositions[tid].position.y;
	dev_vbo_data[15 * tid + 2] = *(fishModel->vertices + 2) + fishPositions[tid].position.z;

	dev_vbo_data[15 * tid + 3] = *(fishModel->vertices + 3) + fishPositions[tid].position.x;
	dev_vbo_data[15 * tid + 4] = *(fishModel->vertices + 4) + fishPositions[tid].position.y;
	dev_vbo_data[15 * tid + 5] = *(fishModel->vertices + 5) + fishPositions[tid].position.z;

	dev_vbo_data[15 * tid + 6] = *(fishModel->vertices + 6) + fishPositions[tid].position.x;
	dev_vbo_data[15 * tid + 7] = *(fishModel->vertices + 7) + fishPositions[tid].position.y;
	dev_vbo_data[15 * tid + 8] = *(fishModel->vertices + 8) + fishPositions[tid].position.z;

	dev_vbo_data[15 * tid + 9] = *(fishModel->vertices + 9) + fishPositions[tid].position.x;
	dev_vbo_data[15 * tid + 10] = *(fishModel->vertices + 10) + fishPositions[tid].position.y;
	dev_vbo_data[15 * tid + 11] = *(fishModel->vertices + 11) + fishPositions[tid].position.z;

	dev_vbo_data[15 * tid + 12] = *(fishModel->vertices + 12) + fishPositions[tid].position.x;
	dev_vbo_data[15 * tid + 13] = *(fishModel->vertices + 13) + fishPositions[tid].position.y;
	dev_vbo_data[15 * tid + 14] = *(fishModel->vertices + 14) + fishPositions[tid].position.z;*/
}

void FishesModel::render(Shader* shader, float* fishes_positions, float* dev_fishes_directions, 
	const glm::mat4& view, const glm::mat4& projection)
{
	

	// compute model matrix for each matrix, then multiply fishModel with each of those matrices and store in vertices
	validateCudaStatus(cudaMemcpy(dev_fishes_positions, fishes_positions, 3 * numberOfFishes * sizeof(float), cudaMemcpyHostToDevice));


	setFishesVertices<<<numberOfBlocks, 1000>>>(dev_vbo_data, dev_fishes_positions, dev_fishes_directions, dev_fish_model);
	
	validateCudaStatus(cudaGetLastError());
	validateCudaStatus(cudaDeviceSynchronize());

	
	

	shader->use();
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);


	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 18 /** numberOfFishes*/);
}


