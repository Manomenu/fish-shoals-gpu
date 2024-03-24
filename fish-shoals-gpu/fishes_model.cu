#include "fishes_model.cuh"

FishesModel::FishesModel(FishesModelCreateInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	numberOfBlocks = (numberOfFishes + MAX_THREADS - 1) / MAX_THREADS;
	fish = new FishModel();

	glCreateVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glCreateBuffers(1, &fish->VBO);
	glBindBuffer(GL_ARRAY_BUFFER, fish->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fish->vertices), fish->vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glCreateBuffers(1, &fish->EBO);	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fish->EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(fish->indices), fish->indices, GL_STATIC_DRAW);

	// https://learnopengl.com/Advanced-OpenGL/Instancing
	glCreateBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(models), models, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(0 * sizeof(glm::vec4)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(1 * sizeof(glm::vec4)));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));
	glVertexAttribDivisor(1, 1);
	glVertexAttribDivisor(2, 1);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);

	#ifdef CPU
	#else
	validateCudaStatus(cudaGraphicsGLRegisterBuffer(&resource_vbo, VBO, cudaGraphicsMapFlagsWriteDiscard));
	#endif
	
}

FishesModel::~FishesModel()
{
	glDeleteBuffers(1, &fish->EBO);
	glDeleteBuffers(1, &fish->VBO);
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
	#ifdef CPU
	#else
	validateCudaStatus(cudaGraphicsUnregisterResource(resource_vbo));
	#endif
	delete fish;
}

__global__ void setModelsKernel(glm::mat4* models, struct cudaSOA soa)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= FISH_COUNT)
		return;

	glm::vec3 v = glm::normalize(soa.velocities[tid]);
	glm::mat4 rotate = glm::toMat4(glm::rotation(glm::vec3(0, 1, 0), v));

	models[tid] = glm::translate(glm::mat4(1), soa.positions[tid]) * rotate;
}

void FishesModel::setModels(
	glm::mat4* models,
	std::vector<glm::vec3>& positions,
	std::vector<glm::vec3>& velocities)
{
	for (int i = 0; i < FISH_COUNT; ++i)
	{
		glm::vec3 v = glm::normalize(velocities[i]);
		glm::mat4 rotate = glm::toMat4(glm::rotation(glm::vec3(0, 1, 0), v));

		models[i] = glm::translate(glm::mat4(1), positions[i]) * rotate;
	}
}

void FishesModel::render(Shader* shader, Fishes* fishes, 
	const glm::mat4& view, const glm::mat4& projection)
{
	// cleaning buffer from trash
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * FISH_COUNT, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// set model matrices
	
	#ifdef CPU
	setModels(models, fishes->positions, fishes->velocities);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(models), &(models)[0], GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	#else
	validateCudaStatus(cudaGraphicsMapResources(1, &resource_vbo));
	validateCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_models, 0, resource_vbo));
	setModelsKernel << <numberOfBlocks, MAX_THREADS >> > (dev_models, fishes->dev_soa);
	validateCudaStatus(cudaPeekAtLastError());
	validateCudaStatus(cudaDeviceSynchronize());
	validateCudaStatus(cudaGraphicsUnmapResources(1, &resource_vbo));
	#endif
	
	
	shader->use();
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);

	glBindVertexArray(VAO);
	glDrawElementsInstanced(GL_TRIANGLES, fish->vertexCount, GL_UNSIGNED_INT, 0, FISH_COUNT);
}


