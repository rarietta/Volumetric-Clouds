// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float volumeIntersectionTest(cameraData cam, ray r, glm::vec3& intersectionPoint);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnGeom(staticGeom geom, float randomSeed);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float geomIntersectionTest(staticGeom geom, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	if (geom.type == CUBE) return boxIntersectionTest(geom, r, intersectionPoint, normal);
	else if (geom.type == SPHERE) return sphereIntersectionTest(geom, r, intersectionPoint, normal);
	return (float)-1.0;
}

//Volume intersection test, return -1 if no intersection, otherwise distance to intersection
__host__ __device__ glm::vec3 getLocalVoxelPosition(glm::vec3 p, volume V)
{	
	float px = (p.x / V.xyzc.x) - 0.5f;
	float py = (p.y / V.xyzc.y) - 0.5f;
	float pz = (p.z / V.xyzc.z) - 0.5f;

	glm::vec3 pos(px, py, pz);
	return pos;
}

//Volume intersection test, return -1 if no intersection, otherwise distance to intersection
__host__ __device__ int getVoxelIndex(glm::vec3 P, volume V){	

	// multiply by inverse transform
	glm::vec3 Pt = multiplyMV(V.inverseTransform, glm::vec4(P,1.0f));

	float error = 0.75e-3;
	if ((Pt.x >= (-0.5 - error)) && (Pt.x <= (0.5 + error)) && 
		(Pt.y >= (-0.5 - error)) && (Pt.y <= (0.5 + error)) && 
		(Pt.z >= (-0.5 - error)) && (Pt.z <= (0.5 + error)))
	{
		int x = floor((Pt.x + 0.5f + error) * V.xyzc.x);
		int y = floor((Pt.y + 0.5f + error) * V.xyzc.y);
		int z = floor((Pt.z + 0.5f + error) * V.xyzc.z);

		return x*V.xyzc.y*V.xyzc.z + y*V.xyzc.z + z;
	}

	return -1;
}

//Volume intersection test, return -1 if no intersection, otherwise distance to intersection
__host__ __device__ float volumeIntersectionTest(volume vol, ray r, glm::vec3& intersectionPoint){
	
	if (getVoxelIndex(r.origin, vol) >= 0)
	{
		intersectionPoint = r.origin;
		return 0.0f;
	}

	glm::vec3 ro = multiplyMV(vol.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(vol.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	glm::vec3 faceNormals[6];
	glm::vec3 faceCenters[6];
	faceNormals[0] = glm::vec3(0,0,-1); faceCenters[0] = glm::vec3(0,0,-0.5);
	faceNormals[1] = glm::vec3(0,0, 1); faceCenters[1] = glm::vec3(0,0, 0.5);
	faceNormals[2] = glm::vec3(0,-1,0); faceCenters[2] = glm::vec3(0,-0.5,0);
	faceNormals[3] = glm::vec3(0, 1,0); faceCenters[3] = glm::vec3(0, 0.5,0);
	faceNormals[4] = glm::vec3(-1,0,0); faceCenters[4] = glm::vec3(-0.5,0,0);
	faceNormals[5] = glm::vec3( 1,0,0); faceCenters[5] = glm::vec3( 0.5,0,0);

	// closest discovered intersection
	float min_t = -1.0;
	int min_i = 6;
  
	// find intersection of ray with each plane of the box
	for (unsigned int i = 0; i < 6; i++) {
		glm::vec3 normal = faceNormals[i];
		glm::vec3 center = faceCenters[i];

		float t = glm::dot((center - rt.origin), normal) / glm::dot(rt.direction, normal);
	  
		// continue if intersection is behind camera
		if (t <= 0)
			continue;
	  
		// if t is greater than the closest found intersection, skip it
		if ((min_t > 0.0) && (t > min_t))
			continue;
	  
		// check to see if the point found is within
		// the edges defined by the face
		glm::vec3 P = getPointOnRay(rt,t);
		float error = 0.75e-3;
		if ((P.x >= (-0.5 - error)) && (P.x <= (0.5 + error)) && 
			(P.y >= (-0.5 - error)) && (P.y <= (0.5 + error)) && 
			(P.z >= (-0.5 - error)) && (P.z <= (0.5 + error)))
		{
			min_t = t;
			min_i = i;
		}
	}
  
	if (min_t < 0)
		return (float) -1.0;
  
	else {
		glm::vec3 realIntersectionPoint = multiplyMV(vol.transform, glm::vec4(getPointOnRay(rt, min_t), 1.0));
		glm::vec3 realNormal = glm::normalize(multiplyMV(vol.transform, glm::vec4(faceNormals[min_i],0.0f)));
		intersectionPoint = realIntersectionPoint;
		return glm::length(r.origin - realIntersectionPoint);
	}
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	        
  //if (box.inverseTransform.z.z == 0.01f) return 1.0;

  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;

  glm::vec3 faceNormals[6];
  glm::vec3 faceCenters[6];
  faceNormals[0] = glm::vec3(0,0,-1); faceCenters[0] = glm::vec3(0,0,-0.5);
  faceNormals[1] = glm::vec3(0,0, 1); faceCenters[1] = glm::vec3(0,0, 0.5);
  faceNormals[2] = glm::vec3(0,-1,0); faceCenters[2] = glm::vec3(0,-0.5,0);
  faceNormals[3] = glm::vec3(0, 1,0); faceCenters[3] = glm::vec3(0, 0.5,0);
  faceNormals[4] = glm::vec3(-1,0,0); faceCenters[4] = glm::vec3(-0.5,0,0);
  faceNormals[5] = glm::vec3( 1,0,0); faceCenters[5] = glm::vec3( 0.5,0,0);

  // closest discovered intersection
  float min_t = -1.0;
  int min_i = 6;
  
  // find intersection of ray with each plane of the box
  for (unsigned int i = 0; i < 6; i++) {
	  glm::vec3 normal = faceNormals[i];
	  glm::vec3 center = faceCenters[i];

	  float t = glm::dot((center - rt.origin), normal) / glm::dot(rt.direction, normal);
	  
	  // continue if intersection is behind camera
	  if (t <= 0)
		  continue;
	  
	  // if t is greater than the closest found intersection, skip it
	  if ((min_t > 0.0) && (t >= min_t))
		  continue;
	  
	  // check to see if the point found is within
	  // the edges defined by the face
	  glm::vec3 P = getPointOnRay(rt,t);
	  float error = 0.75e-3;
	  if ((P.x >= (-0.5 - error)) && (P.x <= (0.5 + error)) && 
		  (P.y >= (-0.5 - error)) && (P.y <= (0.5 + error)) && 
		  (P.z >= (-0.5 - error)) && (P.z <= (0.5 + error)))
	  {
		  min_t = t;
		  min_i = i;
	  }
  }
  
  if (min_t < 0)
	return (float) -1.0;
  
  else {
	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, min_t), 1.0));
	glm::vec3 realNormal = glm::normalize(multiplyMV(box.transform, glm::vec4(faceNormals[min_i],0.0f)));
	intersectionPoint = realIntersectionPoint;
	normal = realNormal;
        
	return glm::length(r.origin - realIntersectionPoint);
  }
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return (float) -1.0;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

__host__ __device__ glm::vec3 getRandomPointOnGeom(staticGeom geom, float randomSeed){
	if (geom.type == SPHERE)    { return getRandomPointOnSphere(geom, randomSeed); }
	else if (geom.type == CUBE) { return getRandomPointOnCube(geom, randomSeed);   }
	else						{ return glm::vec3(0.0f, 0.0f, 0.0f);			   }
}

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(-0.5,0.5);

	float x = (float)u01(rng);
	float y = (float)u01(rng);
	float z = (float)u01(rng);

	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::normalize(glm::vec4(x,y,z,1.0f)));
	return randPoint;
}

#endif


