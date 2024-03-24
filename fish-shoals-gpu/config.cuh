#pragma once

#pragma region General_includes

#include "cuda_runtime.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <sstream>
#include <fstream>
#include <string>
#include "cuda_utils.cuh"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <algorithm>
#include <execution>
#include <random>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#pragma endregion

#pragma region Configuration_constants

const float FAR = 300.0f;
const float NEAR = 0.1f;
const float FOVY = 45.0f;
#define BACKGROUND_COLOR 0.5f, 0.1f, 0.3f, 1.0f
#define AQUARIUM_LEN 10.0f
#define AQUARIUM_SIZE AQUARIUM_LEN, AQUARIUM_LEN, AQUARIUM_LEN
#define FISH_RENDER_H 0.06f
#define FISH_RENDER_A 0.02f
#define FISH_COUNT 50000
#define MAX_THREADS 1024

#pragma endregion

#pragma region Data_types_definitions

struct fishesGrid {
	int* cells;
	int* fishesIDs;
	int* starts;
	int* ends;
};

struct cudaSOA {
	glm::vec3* positions;
	glm::vec3* positions_P;
	glm::vec3* velocities;
	glm::vec3* velocities_P;
	fishesGrid grid;
};

struct fishData {
	glm::vec3 separation_factor = glm::vec3(0);
	glm::vec3 alignment_factor = glm::vec3(0);
	glm::vec3 cohesion_factor = glm::vec3(0);
	glm::vec3 velocity = glm::vec3(0);
	glm::vec3 position = glm::vec3(0);
	int numberOfNeighbours = 0;
};

struct image {
	unsigned char* pixels;
	int width, height, channels;
};

#pragma endregion








