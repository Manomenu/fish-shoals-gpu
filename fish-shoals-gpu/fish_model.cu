#include "fish_model.cuh"

FishModel::FishModel()
{
	float sqrt3 = glm::sqrt(3);
	std::vector<float> vertices =
	{
		h, 0, 0,
		0, 0, a * sqrt3 / 3,
		0, -a / 2, -a * sqrt3 / 6,
		0,	a / 2, -a * sqrt3 / 6,
	};
	std::copy(vertices.begin(), vertices.end(), this->vertices);

	std::vector<int> indices = {
		0, 1, 2,
		0, 2, 3,
		0, 1, 3,
		1, 2, 3
	};
	std::copy(indices.begin(), indices.end(), this->indices);
}