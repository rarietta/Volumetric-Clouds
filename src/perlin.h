//added the function that lets you actually get 3 dimensional noise - Cory Boatright
#ifndef PERLIN_H
#define PERLIN_H

#include <stdlib.h>
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

#define SAMPLE_SIZE 1024
#define B SAMPLE_SIZE
#define BM (SAMPLE_SIZE-1)

#define N1 0x1000
#define NP 12   /* 2^N */
#define NM 0xfff

#define setup(i,b0,b1,r0,r1)\
	t = vec[i] + N1;\
	b0 = ((int)t) & BM;\
	b1 = (b0+1) & BM;\
	r0 = t - (int)t;\
	r1 = r0 - 1.0f;

class Perlin {
public:
  __host__ __device__ Perlin(int octaves,float freq,float amp,int seed);
  __host__ __device__ float Get(glm::vec3 vec);
  __host__ __device__ void  init_perlin(int n,float p);
  __host__ __device__ float perlin_noise_3D(glm::vec3 vec);
  __host__ __device__ float noise3(glm::vec3 vec);
  __host__ __device__ void  init(void);

  int   mOctaves;
  float mFrequency;
  float mAmplitude;
  int   mSeed;

  int p[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  glm::vec3 g3[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  bool mStart;
};

__host__ __device__ inline float randomf(int seed){
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<float> u01(0,RAND_MAX);
  return (float)u01(rng);
}

__host__ __device__ inline float s_curve(float t) {
	return (t * t * (3.0f - 2.0f * t));
}

__host__ __device__ inline float pLerp(float t, float a, float b) {
	return (a + t * (b - a));
}

__host__ __device__ inline float Perlin::Get(glm::vec3(vec)) {
	//return 0.1f;
	return perlin_noise_3D(vec);
}

__host__ __device__ inline float Perlin::noise3(glm::vec3 vec) {

	//return 1.0;
	glm::vec3 q;
	int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
	float rx0, rx1, ry0, ry1, rz0, rz1, sy, sz, a, b, c, d, t, u, v;
	int i, j;
	
	setup(0, bx0,bx1, rx0,rx1);
	setup(1, by0,by1, ry0,ry1);
	setup(2, bz0,bz1, rz0,rz1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];
	
	t  = s_curve(rx0);
	sy = s_curve(ry0);
	sz = s_curve(rz0);
	
    #define at3(rx,ry,rz) ( rx * q[0] + ry * q[1] + rz * q[2] )

	q = g3[b00 + bz0] ; u = at3(rx0,ry0,rz0);
	q = g3[b10 + bz0] ; v = at3(rx1,ry0,rz0);
	a = pLerp(t, u, v);

	q = g3[b01 + bz0] ; u = at3(rx0,ry1,rz0);
	q = g3[b11 + bz0] ; v = at3(rx1,ry1,rz0);
	b = pLerp(t, u, v);

	c = pLerp(sy, a, b);

	q = g3[b00 + bz1] ; u = at3(rx0,ry0,rz1);
	q = g3[b10 + bz1] ; v = at3(rx1,ry0,rz1);
	a = pLerp(t, u, v);

	q = g3[b01 + bz1] ; u = at3(rx0,ry1,rz1);
	q = g3[b11 + bz1] ; v = at3(rx1,ry1,rz1);
	b = pLerp(t, u, v);

	d = pLerp(sy, a, b);
	
	return pLerp(sz, c, d);
}

__host__ __device__ inline float Perlin::perlin_noise_3D(glm::vec3 vec)
{	
	int   terms  = 12;//mOctaves;
	float freq   = 3.0f;//mFrequency;
	float amp    = 8.0f;//mAmplitude;
	float result = 0.0f;

	glm::vec3 v = vec;
	v *= freq;
	
	for(int i = 0; i < terms; i++) {
		result += noise3(v) * amp;
		v *= 2.0f;
		amp *= 0.5f;
	}
	
	return result;
}

__host__ __device__ inline Perlin::Perlin(int octaves,float freq,float amp,int seed) {

	// everything is hard 
	// coded for now

	mOctaves	= octaves;
	mFrequency  = freq;
	mAmplitude  = amp;
	mSeed		= 1234;//seed;
	mStart		= true;

	// init
	int i, j, k;

	for (i = 0 ; i < B ; i++) {
		p[i] = i;
		for (j = 0 ; j < 3 ; j++) {
			g3[i][j] = (float)(((int)randomf(mSeed) % (B + B)) - B) / B;
		}
		glm::normalize(g3[i]);
	}

	while (--i) {
		k = p[i];
		p[i] = p[j = (int)randomf(mSeed) % B];
		p[j] = k;
	}

	for (i = 0 ; i < B + 2 ; i++) {
		p[B + i] = p[i];
		for (j = 0 ; j < 3 ; j++) {
			g3[B + i][j] = g3[i][j];
		}
	}
}

#endif