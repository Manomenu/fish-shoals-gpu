#include "pyramid_model.h"

PyramidModel::PyramidModel(PyramidModelCreateInfo* createInfo)
{
	float h = createInfo->size.x / 2.0f;
	float a = createInfo->size.y;

	vertices = {

		// Vertex Positions
		0.0f, h, 0.0f,		// 0
		-a, -h, a, 		// 1
		a, -h, a, 		// 2
		
		0.0f, h, 0.0f,
		a, -h, a,
		a, -h, -a,	// 3

		0.0f, h, 0.0f,
		a, -h, -a,
		-a, -h, -a, 	// 4

		0.0f, h, 0.0f,		// 0
		-a, -h, a, 		// 1
		-a, -h, -a, 	// 4

		-a, -h, a, 		// 1
		a, -h, a, 		// 2
		-a, -h, -a, 	// 4

		a, -h, a, 		// 2
		a, -h, -a,	// 3
		-a, -h, -a, 	// 4
	};

	vertexCount = vertices.size() / 3;
	//glCreateBuffers(1, &VBO);
	//glCreateVertexArrays(1, &VAO);
	//glVertexArrayVertexBuffer(VAO, 0, VBO, 0, 3 * sizeof(float));
	//glNamedBufferStorage(VBO, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_STORAGE_BIT);
	//glEnableVertexArrayAttrib(VAO, 0);
	//glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
	//glVertexArrayAttribBinding(VAO, 0, 0);
}

PyramidModel::~PyramidModel()
{
	//glDeleteBuffers(1, &VBO);
	//glDeleteVertexArrays(1, &VAO);
}

//void PyramidModel::render(FishTransformation* fish, Shader* shader)
//{
//	glm::mat4 model_transform{ glm::mat4(1.0f) };
//	model_transform = glm::translate(model_transform, fish->position);
//	shader->setMat4("model", model_transform);
//
//	shader->use();
//
//	glBindVertexArray(VAO);
//	glDrawArrays(GL_TRIANGLES, 0, vertexCount);
//}
