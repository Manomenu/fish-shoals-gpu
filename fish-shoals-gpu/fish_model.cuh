#include "config.cuh"


struct FishModel
{
public:
	constexpr static int vertexCount = 12;
	constexpr static int indicesCount = 12;

	float h = FISH_RENDER_H;
	float a = FISH_RENDER_A;
	unsigned int VBO = 0, EBO = 0;
	float vertices[vertexCount];
	int indices[vertexCount];

	FishModel();

};