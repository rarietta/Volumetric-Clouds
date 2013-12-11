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

__host__ __device__ float generateRandomFloatFromSeed(int index, float seed){
  thrust::default_random_engine rng(hash(index*seed));
  thrust::uniform_real_distribution<float> u01(0,1);

  return (float)u01(rng);
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

		if(color.x>255)
			color.x = 255;
		else if (color.x<0)
			color.x = 0;
		
		if(color.y>255)
			color.y = 255;
		else if (color.y<0)
			color.y = 0;
		
		if(color.z>255)
			color.z = 255;
		else if (color.z<0)
			color.z = 0;
      
		// Each thread writes one pixel location in the texture (textel)
		PBOpos[index].w = 1.0;
		PBOpos[index].x = color.x;
		PBOpos[index].y = color.y;
		PBOpos[index].z = color.z;
	}
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void voxelizeVolumeWithNoise(int index, volume* volumes, Perlin* perlin1, Perlin* perlin2, int timestep)
{
	// identify current volume
	volume V = volumes[index];

	// get index of voxel within volume
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (blockIdx.z * blockDim.z) + threadIdx.z;
	int voxelIndex = x*V.xyzc.y*V.xyzc.z + y*V.xyzc.z + z;
	
	// find distance from voxel to center of volume
	glm::vec3 localPosition3D = getLocalVoxelPosition(glm::vec3((float)x, (float)y, (float)z), V);
	float length = glm::distance(localPosition3D, glm::vec3(0.0f, 0.0f, 0.0f));
	
	// get random number
	//float modifier = generateRandomFloatFromSeed(voxelIndex, perlin1->mSeed) * 0.5 - 0.25;
	float modifier = 0.0f;

	if ((length < 0.5f + modifier) && (voxelIndex < V.xyzc.x*V.xyzc.y*V.xyzc.z) && (localPosition3D.y < 0.2)) {
		float p1 = (perlin1->Get(multiplyMV(V.transform, glm::vec4(localPosition3D, 1.0))) + (1.0 - (length / (0.5f + modifier)))) * ((0.5f + modifier) - length);
		float p2 = (perlin2->Get(multiplyMV(V.transform, glm::vec4(localPosition3D, 1.0))) + (1.0 - (length / (0.5f + modifier)))) * ((0.5f + modifier) - length);
		V.voxels[voxelIndex].density = max(glm::mix(p1, p2, (float)(timestep%20) / 20.0f), 0.0f);
	}
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, glm::vec3* colors, light* lights, int numberOfLights,
							material* materials, volume* volumes, int numberOfVolumes, float iterations)
{
	// Find index of pixel and create empty color vector
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	// Get initial ray from camera through this position
	ray currentRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  
	// Return values for the intersection test
	glm::vec3 intersection_point;

	// constant attenuation for transmission
	float k = 0.2f;

	// initialize color along ray to black
	colors[index] = cam.brgb;
	
	for (int v = 0; v < numberOfVolumes; v++)
	{
		// initialize transmission of pixel to 1.0
		float T = 1.0;
	
		glm::vec3 newColor = glm::vec3(0.0f);
		volume V = volumes[v];

		if (volumeIntersectionTest(V, currentRay, intersection_point) > 0.0)
		{
			// initial intersection point on bounding box of volume
			glm::vec3 marchPoint = intersection_point;

			// color of volumetric material 
			glm::vec3 volCol = materials[V.materialid].color;

			// index of initial intersection point in volume density grid
			int voxelIndex = getVoxelIndex(marchPoint, V);

			// recurse through the volume and perform operations
			// while still inside (i.e. point has valid voxel index)
			while (voxelIndex >= 0) {

				// density of voxel at point
				float p = V.voxels[voxelIndex].density;
			
				// transmission value at point evaluated using given function
				float deltaT = exp(-k*V.step*p);
			
				// accumulate transmission along ray
				// and break if below threshold
				T *= min(deltaT, 1.0f);
				if (T < 0.1) break;

				// calculate lighting
				if (deltaT < 1.0f) {
					for (int i = 0; i < numberOfLights; i++)
					{
						// initialize transmission along
						// light ray to zero
						float Q = 1.0;

						// ith scene light
						light L = lights[i];

						// material color scaled by light intensity
						glm::vec3 CF = volCol * L.color;

						// first sampling point along light ray is
						// march point
						glm::vec3 lightPoint = marchPoint;

						// light ray
						glm::vec3 lightDir = glm::normalize(L.position - marchPoint);
				
						// get index of voxel for point along light ray
						int lightVoxelIndex = getVoxelIndex(lightPoint, V);
				
						// recurse along light ray and perform operations
						// while still inside (i.e. point has valid voxel index
						while (lightVoxelIndex >= 0) 
						{
							// density at point along light ray
							float pLight = V.voxels[lightVoxelIndex].density;
					
							// light transmission value at point along light ray
							float deltaQ = exp(-k*V.step*pLight);

							// accumulate opacity of point
							Q *= deltaQ;
							if (Q < 0.05) break;

							// step to next sample point along light ray
							lightPoint += lightDir * V.step;

							// get next voxel index
							lightVoxelIndex = getVoxelIndex(lightPoint, V);
						}
						// accumulate color value
						newColor += (1.0f - deltaT)/k * (CF * T * Q);
						glm::clamp(newColor, 0.0f, 1.0f);
					}
				}
				// increment marching point along ray by step size
				marchPoint += V.step * glm::normalize(currentRay.direction);

				// get new voxel index for next loop
				voxelIndex = getVoxelIndex(marchPoint, V);
			}
		} 
		// blend with background color according to transmission
		glm::clamp(T, 0.0f, 1.0f);
		colors[index] = glm::mix(newColor, colors[index], T);
		colors[index] = glm::clamp(colors[index], 0.0f, 1.0f);
	}
}


// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int timestep, material* materials, int numberOfMaterials, 
					  volume* volumes, int numberOfVolumes, light* lights, int numberOfLights, Perlin* perlin1, Perlin* perlin2)
{
	// set up crucial magic
	int tileSize = 10;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
	//package volumes and send to GPU
	volume* volumeList = new volume[numberOfVolumes];
	for (int i = 0; i < numberOfVolumes; i++) {
		volume newVolume;
		newVolume.isSet			   = volumes[i].isSet;
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

		voxel* cudaVolumeVoxels = NULL;
		int numVoxels = int(newVolume.xyzc.x*newVolume.xyzc.y*newVolume.xyzc.z);
		cudaMalloc((void**)&cudaVolumeVoxels, numVoxels*sizeof(voxel));
		
		voxel* voxels = new voxel[numVoxels];
		for (int v = 0; v < numVoxels; v++) {
			voxels[v].density = volumes[i].voxels[v].density;
			//if (v % 1000 == 0) printf("iteration %d, voxel %d: %f\n", timestep, v, voxels[v].density);
		}
		cudaMemcpy(cudaVolumeVoxels, voxels, numVoxels*sizeof(voxel), cudaMemcpyHostToDevice);
		newVolume.voxels = cudaVolumeVoxels;

		volumeList[i] = newVolume;

		delete voxels;
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
	light* cudalights = NULL;
	cudaMalloc((void**)&cudalights, numberOfLights*sizeof(light));
	cudaMemcpy(cudalights, lightList, numberOfLights*sizeof(light), cudaMemcpyHostToDevice);
  

	//package perlin
	Perlin* cudaperlin1 = NULL;
	cudaMalloc((void**)&cudaperlin1, sizeof(Perlin));
	cudaMemcpy(cudaperlin1, perlin1, sizeof(Perlin), cudaMemcpyHostToDevice);
	Perlin* cudaperlin2 = NULL;
	cudaMalloc((void**)&cudaperlin2, sizeof(Perlin));
	cudaMemcpy(cudaperlin2, perlin2, sizeof(Perlin), cudaMemcpyHostToDevice);
	

	//package camera
	cameraData cam;
	cam.delt = renderCam->delt;
	cam.step = renderCam->step;
	cam.brgb = renderCam->brgb;
	cam.xyzc = renderCam->xyzc;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->position;
	cam.view = renderCam->view;
	cam.up = renderCam->up;
	cam.fov = renderCam->fov;


	// kernel call to populate voxel densities
	for (int i = 0; i < numberOfVolumes; i++) {
		//if (!volumes[i].isSet) {
			dim3 voxelThreadsPerBlock(tileSize, tileSize, tileSize);
			dim3 voxelFullBlocksPerGrid((int)ceil(float(volumes[i].xyzc.x)/float(tileSize)), 
										(int)ceil(float(volumes[i].xyzc.y)/float(tileSize)), 
										(int)ceil(float(volumes[i].xyzc.z)/float(tileSize)));
			voxelizeVolumeWithNoise<<<voxelFullBlocksPerGrid, voxelThreadsPerBlock>>>(i, cudavolumes, cudaperlin1, cudaperlin2, (float)timestep);
		//}
	}
	

	//kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)timestep, cam, cudaimage, cudalights, numberOfLights, cudamaterials, 
		cudavolumes, numberOfVolumes, renderCam->iterations);
  
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  

	//retrieve image from GPU
	cudaMemcpy(renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  
	// save state of volumes
	volume* volumesArr = new volume[numberOfVolumes];
	cudaMemcpy(volumesArr, cudavolumes, numberOfVolumes*sizeof(volume), cudaMemcpyDeviceToHost);

	for (int i = 0; i < numberOfVolumes; i++) {
		if (!volumes[i].isSet) {
			int numVoxels = int(volumes[i].xyzc.x*volumes[i].xyzc.y*volumes[i].xyzc.z);
			cudaMemcpy(volumes[i].voxels, volumesArr[i].voxels, numVoxels*sizeof(voxel), cudaMemcpyDeviceToHost);
			volumes[i].isSet = true;
		}
	}

	//free up stuff, or else we'll leak memory like a madman
	delete lightList;
	for (int i = 0; i < numberOfVolumes; i++) {
		cudaFree( volumeList[i].voxels);
	}
	delete volumeList;
	delete volumesArr;
	delete materialList;
	cudaFree( cudaimage );
	cudaFree( cudalights );
	cudaFree( cudavolumes );
	cudaFree( cudamaterials );
	cudaFree( cudaperlin1 );
	cudaFree( cudaperlin2 );


	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
