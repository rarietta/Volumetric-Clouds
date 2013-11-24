//added the function that lets you actually get 3 dimensional noise - Cory Boatright
#ifndef PERLIN_H_
#define PERLIN_H_

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
  __host__ __device__ float Get(float x,float y);
  __host__ __device__ float Get(float x, float y, float z);

private:
  __host__ __device__ void init_perlin(int n,float p);
  __host__ __device__ float perlin_noise_2D(float vec[2]);
  __host__ __device__ float perlin_noise_3D(float vec[3]);

  __host__ __device__ float noise1(float arg);
  __host__ __device__ float noise2(float vec[2]);
  __host__ __device__ float noise3(float vec[3]);
  __host__ __device__ void normalize2(float v[2]);
  __host__ __device__ void normalize3(float v[3]);
  __host__ __device__ void init(void);

  int mOctaves;
  float mFrequency;
  float mAmplitude;
  int mSeed;

  int p[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  float g3[SAMPLE_SIZE + SAMPLE_SIZE + 2][3];
  float g2[SAMPLE_SIZE + SAMPLE_SIZE + 2][2];
  float g1[SAMPLE_SIZE + SAMPLE_SIZE + 2];
  bool mStart;
};


__host__ __device__ float random(void){
  thrust::default_random_engine rng(1234567);
  thrust::uniform_real_distribution<float> u01(0,RAND_MAX);
  return (float)u01(rng);
}

__host__ __device__ inline float s_curve(float t) {
	return (t * t * (3.0f - 2.0f * t));
}

__host__ __device__ inline float pLerp(float t, float a, float b) {
	return (a + t * (b - a));
}

__host__ __device__ inline float Perlin::Get(float x, float y){
	float vec[2];
	vec[0] = x;
	vec[1] = y;
	return perlin_noise_2D(vec);
};

__host__ __device__ inline float Perlin::Get(float x, float y, float z) {
	float vec[3];
	vec[0] = x;
	vec[1] = y;
	vec[2] = z;
	return perlin_noise_3D(vec);
}

__host__ __device__ inline float Perlin::noise1(float arg) {
	int bx0, bx1;
	float rx0, rx1, sx, t, u, v, vec[1];

	vec[0] = arg;

	if (mStart) {
		random();
		mStart = false;
		init();
	}

	setup(0, bx0,bx1, rx0,rx1);

	sx = s_curve(rx0);

	u = rx0 * g1[p[bx0]];
	v = rx1 * g1[p[bx1]];

	return pLerp(sx, u, v);
}

__host__ __device__ inline float Perlin::noise2(float vec[2]) {
	int bx0, bx1, by0, by1, b00, b10, b01, b11;
	float rx0, rx1, ry0, ry1, *q, sx, sy, a, b, t, u, v;
	int i, j;

	if (mStart) {
		random();
		mStart = false;
		init();
	}

	setup(0,bx0,bx1,rx0,rx1);
	setup(1,by0,by1,ry0,ry1);

	i = p[bx0];
	j = p[bx1];

	b00 = p[i + by0];
	b10 = p[j + by0];
	b01 = p[i + by1];
	b11 = p[j + by1];

	sx = s_curve(rx0);
	sy = s_curve(ry0);

	#define at2(rx,ry) (rx * q[0] + ry * q[1])

	q = g2[b00];
	u = at2(rx0,ry0);
	q = g2[b10];
	v = at2(rx1,ry0);
	a = pLerp(sx, u, v);

	q = g2[b01];
	u = at2(rx0,ry1);
	q = g2[b11];
	v = at2(rx1,ry1);
	b = pLerp(sx, u, v);

	return pLerp(sy, a, b);
}

__host__ __device__ inline float Perlin::noise3(float vec[3]) {
	int bx0, bx1, by0, by1, bz0, bz1, b00, b10, b01, b11;
	float rx0, rx1, ry0, ry1, rz0, rz1, *q, sy, sz, a, b, c, d, t, u, v;
	int i, j;

	if (mStart) {
		random();
		mStart = false;
		init();
	}

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

__host__ __device__ inline void Perlin::normalize2(float v[2]) {
	float s;

	s = (float)sqrt(v[0] * v[0] + v[1] * v[1]);
	s = 1.0f/s;
	v[0] = v[0] * s;
	v[1] = v[1] * s;
}

__host__ __device__ inline void Perlin::normalize3(float v[3]) {
	float s;

	s = (float)sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	s = 1.0f/s;

	v[0] = v[0] * s;
	v[1] = v[1] * s;
	v[2] = v[2] * s;
}

__host__ __device__ inline void Perlin::init(void) {
	int i, j, k;

	for (i = 0 ; i < B ; i++) {
		p[i] = i;
		g1[i] = (float)(((int)random() % (B + B)) - B) / B;
		for (j = 0 ; j < 2 ; j++) {
			g2[i][j] = (float)(((int)random() % (B + B)) - B) / B;
		}
		normalize2(g2[i]);
		for (j = 0 ; j < 3 ; j++) {
			g3[i][j] = (float)(((int)random() % (B + B)) - B) / B;
		}
		normalize3(g3[i]);
	}

	while (--i) {
		k = p[i];
		p[i] = p[j = (int)random() % B];
		p[j] = k;
	}

	for (i = 0 ; i < B + 2 ; i++) {
		p[B + i] = p[i];
		g1[B + i] = g1[i];
		for (j = 0 ; j < 2 ; j++) {
			g2[B + i][j] = g2[i][j];
		}
		for (j = 0 ; j < 3 ; j++) {
			g3[B + i][j] = g3[i][j];
		}
	}
}

__host__ __device__ inline float Perlin::perlin_noise_2D(float vec[2]) {
	int terms = mOctaves;
	float freq = mFrequency;
	float result = 0.0f;
	float amp = mAmplitude;
	
	vec[0]*=mFrequency;
	vec[1]*=mFrequency;
	
	for(int i=0; i < terms; i++) {
		result += noise2(vec)*amp;
		vec[0] *= 2.0f;
		vec[1] *= 2.0f;
		amp*=0.5f;
	}
	
	return result;
}

__host__ __device__ inline float Perlin::perlin_noise_3D(float vec[3]) {
	int terms = mOctaves;
	float freq = mFrequency;
	float result = 0.0f;
	float amp = mAmplitude;

	vec[0] *= mFrequency;
	vec[1] *= mFrequency;
	vec[2] *= mFrequency;

	for(int i=0; i<terms; i++ ) {
		result += noise3(vec)*amp;
		vec[0] *= 2.0f;
		vec[1] *= 2.0f;
		vec[2] *= 2.0f;
		amp*=0.5f;
	}

	return result;
}

__host__ __device__ inline Perlin::Perlin(int octaves,float freq,float amp,int seed) {
	mOctaves = octaves;
	mFrequency = freq;
	mAmplitude = amp;
	mSeed = seed;
	mStart = true;
}

#endif