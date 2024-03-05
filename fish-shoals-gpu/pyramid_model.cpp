#include "pyramid_model.h"

PyramidModel::PyramidModel(PyramidModelCreateInfo* createInfo)
{
	float h = createInfo->size.x / 2.0f;
	float a = createInfo->size.y;
	vertexCount = 15;

	float verticesData[] = {
		-a, -h, a, //0
		a, -h, a, //1
		a, -h, -a, //2
		-a, -h, -a, //3
		0.0, h, 0.0 //4
	};

	for (int i = 0; i < vertexCount; i++)
	{
		vertices[i] = verticesData[i];
	}
}

PyramidModel::~PyramidModel()
{
}
