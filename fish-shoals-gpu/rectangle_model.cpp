#include "rectangle_model.h"

RectangleModel::RectangleModel(RectangleModelCreateInfo* createInfo) {

	float l = createInfo->size.x / 2;
	float w = createInfo->size.y / 2;
	float h = createInfo->size.z / 2;

	//Make Cube
	//x,y,z,s,t
	vertices = { {
		-l, -w, -h, 0.0f, 0.0f, // bottom
		 l, -w, -h, 1.0f, 0.0f,
		 l,  w, -h, 1.0f, 1.0f,

		 l,  w, -h, 1.0f, 1,
		-l,  w, -h, 0.0f, 1,
		-l, -w, -h, 0.0f, 0,

		-l, -w,  h, 0.0f, 0.0f, //top
		 l, -w,  h, 1.0f, 0.0f,
		 l,  w,  h, 1.0f, 1.0f,

		 l,  w,  h, 1.0f, 1.0f,
		-l,  w,  h, 0.0f, 1.0f,
		-l, -w,  h, 0.0f, 0.0f,

		-l,  w,  h, 1.0f, 0.0f, //left
		-l,  w, -h, 1.0f, 1.0f,
		-l, -w, -h, 0.0f, 1.0f,

		-l, -w, -h, 0.0f, 1.0f,
		-l, -w,  h, 0.0f, 0.0f,
		-l,  w,  h, 1.0f, 0.0f,

		 l,  w,  h, 1.0f, 0.0f, //right
		 l,  w, -h, 1.0f, 1.0f,
		 l, -w, -h, 0.0f, 1.0f,

		 l, -w, -h, 0.0f, 1.0f,
		 l, -w,  h, 0.0f, 0.0f,
		 l,  w,  h, 1.0f, 0.0f,

		-l, -w, -h, 0.0f, 1.0f, //back
		 l, -w, -h, 1.0f, 1.0f,
		 l, -w,  h, 1.0f, 0.0f,

		 l, -w,  h, 1.0f, 0.0f,
		-l, -w,  h, 0.0f, 0.0f,
		-l, -w, -h, 0.0f, 1.0f,

		-l,  w, -h, 0.0f, 1.0f, //front
		 l,  w, -h, 1.0f, 1.0f,
		 l,  w,  h, 1.0f, 0.0f,

		 l,  w,  h, 1.0f, 0.0f,
		-l,  w,  h, 0.0f, 0.0f,
		-l,  w, -h, 0.0f, 1.0f
	} };
	vertexCount = vertices.size() / 5;
	glCreateBuffers(1, &VBO);
	glCreateVertexArrays(1, &VAO);
	glVertexArrayVertexBuffer(VAO, 0, VBO, 0, 5 * sizeof(float));
	glNamedBufferStorage(VBO, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_STORAGE_BIT);
	glEnableVertexArrayAttrib(VAO, 0);
	glEnableVertexArrayAttrib(VAO, 1);
	glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribFormat(VAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
	glVertexArrayAttribBinding(VAO, 0, 0);
	glVertexArrayAttribBinding(VAO, 1, 0);
}

RectangleModel::~RectangleModel() {
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}

void RectangleModel::render(Cube* cube, Shader* shader, bool shouldFill, 
	const glm::mat4& view, const glm::mat4& projection)
{
	glm::mat4 model_transform{ glm::mat4(1.0f) };
	model_transform = glm::translate(model_transform, cube->position);

	shader->use();
	shader->setMat4("model", model_transform);
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);

	glBindVertexArray(VAO);
	if (!shouldFill)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawArrays(GL_TRIANGLES, 0, vertexCount);
	if (!shouldFill)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
