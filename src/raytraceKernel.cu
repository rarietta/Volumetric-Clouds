// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include "perlin.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){

  //establish "right" camera direction
  glm::normalize(eye); glm::normalize(view);
  glm::vec3 right = glm::normalize(glm::cross(up, view));
  
  // calculate P1 and P2 in both x and y directions
  glm::vec3 image_center = eye + view;
  glm::vec3 P1_X = image_center - tan((float)4.0*fov.x)*right;
  glm::vec3 P2_X = image_center + tan((float)4.0*fov.x)*right;
  glm::vec3 P1_Y = image_center - tan((float)4.0*fov.y)*up;
  glm::vec3 P2_Y = image_center + tan((float)4.0*fov.y)*up;
  
  glm::vec3 bottom_left  = P1_X + (P1_Y - image_center);
  glm::vec3 bottom_right = P2_X + (P1_Y - image_center);
  glm::vec3 top_left     = P1_X + (P2_Y - image_center);

  glm::vec3 imgRight = bottom_right - bottom_left;
  glm::vec3 imgUp    = top_left - bottom_left;

  // supersample the pixels by taking a randomly offset ray in each iteration
  glm::vec3 random_offset = generateRandomNumberFromThread(resolution, time, x, y);
  float x_offset = random_offset.x;
  float y_offset = random_offset.y;
  glm::vec3 img_point = bottom_left + ((float)x + x_offset)/(float)resolution.x*imgRight + ((float)y + y_offset)/(float)resolution.y*imgUp;
  glm::vec3 direction = glm::normalize(img_point - eye); 

  // return value
  ray r; r.origin = eye; r.direction = direction;
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, glm::vec3* colors, staticGeom* lights, int numberOfLights,
							material* materials, volume* volumes, int numberOfVolumes, float iterations)
{
	// Find index of pixel and create empty color vector
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	glm::vec3 newColor = cam.brgb;

	// Get initial ray from camera through this position
	ray currentRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		
	ray reflectionRay;
	int currentDepth = 0;
	bool reflect = false;
	glm::vec3 currentSpecCoeff(1.0f, 1.0f, 1.0f);
  
	// Return values for the intersection test
	glm::vec3 intersection_point;
	glm::vec3 intersection_normal;
	material  intersection_mtl;


	// if voxel grid is intersected, march through it
	if (volumeIntersectionTest(volumes[0], currentRay, intersection_point) > 0.0)
	{
		glm::vec3 marchPoint = intersection_point;

		glm::vec3 voxelIndex = getVoxelIndex(marchPoint, volumes[0]);
		while (voxelIndex.x >= 0) {
			//float xCol = voxelIndex.x / volumes[0].xyzc.x;
			//float yCol = voxelIndex.y / volumes[0].xyzc.y;
			//float zCol = voxelIndex.z / volumes[0].xyzc.z;
			//newColor = glm::mix(newColor, glm::vec3(xCol,yCol,zCol), 0.5f);
			newColor = glm::mix(newColor, materials[volumes[0].materialid].color, volumes[0].densities[0]);

			marchPoint += volumes[0].step * glm::normalize(currentRay.direction);
			voxelIndex = getVoxelIndex(marchPoint, volumes[0]);
		}
	} 

	/*
	do {
		// Find the closest geometry intersection along the ray
		float t;
		float min_t = -1.0;
		for (int i = 0; i < numberOfGeoms; i++) {
			staticGeom geom = geoms[i];
			t = geomIntersectionTest(geom, currentRay, intersection_point, intersection_normal);
			if ((t > 0.0) && (t < min_t || min_t < 0.0)) {
				min_t = t;
				intersection_mtl = materials[geom.materialid];
			}
		}
		
		// find reflected ray if one exists
		if (intersection_mtl.hasReflective) {
			reflect = true;
			glm::vec3 rd = calculateReflectionDirection(intersection_normal, currentRay.direction);
			glm::vec3 ro = glm::vec3(intersection_point);
			reflectionRay.direction = rd; reflectionRay.origin = ro;
		}
		else { reflect = false; }
		
		// Find and clamp diffuse contribution at point
		glm::vec3 phong = computePhongTotal(currentRay, intersection_point, intersection_normal, intersection_mtl, 
											lights, numberOfLights, geoms, numberOfGeoms, materials, (float)time);
		if (phong.x > 1.0f) { phong.x = 1.0f; } else if (phong.x < 0.0f) { phong.x = 0.0f; }
		if (phong.y > 1.0f) { phong.y = 1.0f; } else if (phong.y < 0.0f) { phong.y = 0.0f; }
		if (phong.z > 1.0f) { phong.z = 1.0f; } else if (phong.z < 0.0f) { phong.z = 0.0f; }
		newColor += (currentSpecCoeff * phong);

		currentDepth++;
		currentRay.origin = reflectionRay.origin;
		currentRay.direction = reflectionRay.direction;
		currentSpecCoeff *= intersection_mtl.specularColor;
	}
	while (reflect && (currentDepth < traceDepth));
	*/

	//if (newColor.x > 1.0f) { newColor.x = 1.0f; } else if (newColor.x < 0.0f) { newColor.x = 0.0f; }
	//if (newColor.y > 1.0f) { newColor.y = 1.0f; } else if (newColor.y < 0.0f) { newColor.y = 0.0f; }
	//if (newColor.z > 1.0f) { newColor.z = 1.0f; } else if (newColor.z < 0.0f) { newColor.z = 0.0f; }
	//if((x<=resolution.x && y<=resolution.y))
	//{
		colors[index] += newColor / (float)iterations;
	//}
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int timestep, material* materials, int numberOfMaterials, 
					  volume* volumes, int numberOfVolumes, light* lights, int numberOfLights)
{
	//determines how many bounces the raytracer traces
	int traceDepth = 3;

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
	//package volumes and send to GPU
	int totalVoxels = 0;
	volume* volumeList = new volume[numberOfVolumes];
	for (int i = 0; i < numberOfVolumes; i++) {
		volume newVolume;
		newVolume.volumeid         = volumes[i].volumeid;
		newVolume.materialid       = volumes[i].materialid;
		newVolume.delt             = volumes[i].delt;
		newVolume.step             = volumes[i].step;
		newVolume.xyzc             = volumes[i].xyzc;
		newVolume.translation      = volumes[i].translation;
		newVolume.rotation         = volumes[i].rotation;
		newVolume.scale            = volumes[i].scale;
		newVolume.transform        = volumes[i].transform;
		newVolume.inverseTransform = volumes[i].inverseTransform;

		float* cudaVolumeDensities = NULL;
		int numVoxels = int(newVolume.xyzc.x*newVolume.xyzc.y*newVolume.xyzc.z);
		cudaMalloc((void**)&cudaVolumeDensities, numVoxels*sizeof(float));
		
		float* densities = new float[numVoxels];
		for (int v = 0; v < numVoxels; v++)
			newVolume.densities[v] = volumes[i].densities[v];
		totalVoxels += numVoxels;
		cudaMemcpy(cudaVolumeDensities, densities, numVoxels*sizeof(float), cudaMemcpyHostToDevice);
		newVolume.densities = cudaVolumeDensities;

		volumeList[i] = newVolume;
	}
	volume* cudavolumes = NULL;
	cudaMalloc((void**)&cudavolumes, numberOfVolumes*sizeof(volume));
	cudaMemcpy(cudavolumes, volumeList, numberOfVolumes*sizeof(volume), cudaMemcpyHostToDevice);
  
	//package materials and send to GPU
	material* materialList = new material[numberOfMaterials];
	for (int i=0; i<numberOfMaterials; i++){
		material newMaterial;
		newMaterial.color = materials[i].color;
		materialList[i] = newMaterial;
	}
	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy(cudamaterials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
	// package lights and send to GPU
	light* lightList = new light[numberOfLights];
	for(int i=0; i<numberOfLights; i++){
		light newLight;
		newLight.position = lights[i].position;
		newLight.color = lights[i].color;
		lightList[i] = newLight;
	}
	staticGeom* cudalights = NULL;
	cudaMalloc((void**)&cudalights, numberOfLights*sizeof(staticGeom));
	cudaMemcpy(cudalights, lightList, numberOfLights*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
	//package camera
	cameraData cam;
	cam.delt = renderCam->delt;
	cam.step = renderCam->step;
	cam.brgb = renderCam->brgb;
	cam.xyzc = renderCam->xyzc;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
	
	//kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)timestep, cam, cudaimage, cudalights, numberOfLights, cudamaterials, 
		cudavolumes, numberOfVolumes, renderCam->iterations);
  
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  
	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
	//free up stuff, or else we'll leak memory like a madman
	delete lightList;
	delete volumeList;
	delete materialList;
	cudaFree( cudaimage );
	cudaFree( cudalights );
	cudaFree( cudavolumes );
	cudaFree( cudamaterials );

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
