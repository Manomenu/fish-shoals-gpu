#pragma once
#include "config.cuh"

#define validateCudaStatus(cudaStatus)												\
{																					\
	if (cudaStatus)																	\
	{																				\
		fprintf(stderr, "Error at line %d in %s \n", __LINE__, __FILE__);			\
		fprintf(stderr, "Error code: %s \n", cudaGetErrorName(cudaStatus));			\
		fprintf(stderr, "Error description: %s \n", cudaGetErrorString(cudaStatus));\
		exit(1);																	\
	}																				\
}

#define ERROR(message)																						  \
    {																									      \
        std::cerr << "Error at line " << __LINE__ << " in file " << __FILE__ << ": " << message << std::endl; \
		exit(1);																							  \
    }