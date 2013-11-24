// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct geom {
	enum GEOMTYPE type;
	int objectid;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
};

struct staticGeom {
	enum GEOMTYPE type;
	int objectid;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
};

struct volume {
	int volumeid;
	int materialid;
	float delt;
	float step;
	glm::vec3 xyzc;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	float* densities;
};

struct light {
	int lightid;
	glm::vec3 color;
	glm::vec3 position;
};

struct cameraData {
	float delt;
	float step;
	glm::vec3 brgb;
	glm::vec3 xyzc;
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
};

struct camera {

	// volumetric data for 
	// ray marching
	float delt;					// dimenstion of square of voxel
	float step;					// size of step to be take by ray
	glm::vec3 xyzc;				// dimension of voxel grid

	// intrinsic camera stuff
	glm::vec3* positions;		// eye positions
	glm::vec3* views;			// camera directions
	glm::vec3* ups;				// up vectors
	glm::vec2 fov;				// y-field-of-view

	// render data
	int frames;					// number of frames
	glm::vec3 brgb;				// background color of render
	glm::vec3* image;			// final image
	glm::vec2 resolution;		// resolution of image
	std::string imageName;		// filename to write image to
	unsigned int iterations;	// number of iterations per frame
};

struct material{
	glm::vec3 color;
	//float specularExponent;
	//glm::vec3 specularColor;
	//float hasReflective;
	//float hasRefractive;
	//float indexOfRefraction;
	//float hasScatter;
	//glm::vec3 absorptionCoefficient;
	//float reducedScatterCoefficient;
	//float emittance;
};

#endif //CUDASTRUCTS_H
