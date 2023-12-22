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

#pragma endregion

#pragma region Configuration_constants

const float FAR = 300.0f;
const float NEAR = 0.1f;
const float FOVY = 45.0f;
#define BACKGROUND_COLOR 0.5f, 0.1f, 0.3f, 1.0f
#define AQUARIUM_SIZE 50.0f, 50.0f, 50.0f

#pragma endregion

#pragma region Data_types_definitions

struct image {
	unsigned char* pixels;
	int width, height, channels;
};

#pragma endregion







