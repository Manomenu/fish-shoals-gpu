#include "fishes_model.cuh"

FishesModel::FishesModel(FishesModelCreateInfo* createInfo)
{
	numberOfFishes = createInfo->numberOfFishes;
	vertexCount = createInfo->numberOfFishes * 5;
	vertices = std::vector<float>(vertexCount * 6);

	std::vector<int> indices = std::vector<int>(vertexCount);

	glCreateBuffers(1, &VBO);
	glCreateVertexArrays(1, &VAO);
	glCreateBuffers(1, &EBO);
	glVertexArrayVertexBuffer(VAO, 0, VBO, 0, 6 * sizeof(float));
	glNamedBufferStorage(VBO, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(VAO, EBO);
	glNamedBufferStorage(EBO, indices.size() * sizeof(float), indices.data(), GL_STATIC_DRAW);
	glEnableVertexArrayAttrib(VAO, 0);
	glEnableVertexArrayAttrib(VAO, 1);
	glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribFormat(VAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
	glVertexArrayAttribBinding(VAO, 0, 0);
	glVertexArrayAttribBinding(VAO, 1, 0);

	PyramidModelCreateInfo fishModelInfo;
	fishModelInfo.size = { 1.0f, 0.3f };
	fishModel = new PyramidModel(&fishModelInfo);

	validateCudaStatus(cudaGraphicsGLRegisterBuffer(&resource_vbo, VBO, cudaGraphicsMapFlagsNone));
	validateCudaStatus(cudaGraphicsMapResources(1, &resource_vbo));
	validateCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_vbo_data, &vbo_size, resource_vbo));

	validateCudaStatus(cudaMalloc((void**)&dev_fishes_positions, sizeof(float) * numberOfFishes));
	validateCudaStatus(cudaMalloc((void**)&dev_fish_model, sizeof(PyramidModel)));
	validateCudaStatus(cudaMemcpy(dev_fish_model, fishModel, sizeof(PyramidModel), cudaMemcpyHostToDevice));
}

FishesModel::~FishesModel()
{
	glDeleteBuffers(1, &EBO);
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
	delete fishModel;
	validateCudaStatus(cudaGraphicsUnmapResources(1, &resource_vbo));
	cudaFree(dev_fishes_positions);
}

void FishesModel::render(Shader* shader, FishTransformation* fishes)
{
	// compute model matrix for each matrix, then multiply fishModel with each of those matrices and store in vertices
	validateCudaStatus(cudaMemcpy(dev_fishes_positions, &fishes->position, sizeof(glm::vec3) * numberOfFishes, cudaMemcpyHostToDevice));
	setFishesVertices(dev_vbo_data, dev_fishes_positions, dev_fish_model);

	shader->use();

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}

__global__ void FishesModel::setFishesVertices(float* dev_vbo_data, float* fishes_positions, PyramidModel* fishModel)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	dev_vbo_data[tid] = tid;
}
