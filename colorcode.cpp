#include "stdafx.h"
#include "colorcode.h"
#include <math.h>
#include <stdlib.h>

#define MAXCOLS 60

namespace flowColorer
{
	int ncols = 0;
	int colorwheel[MAXCOLS][3];
}

using namespace flowColorer;


void setcols(int r, int g, int b, int k)
{
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}

void makeColorwheel()
{
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	if (ncols > 0)//already computed
		return;
	
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	//printf("ncols = %d\n", ncols);
	if (ncols > MAXCOLS)
		exit(1);
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
	for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
	for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
	for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
	for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
	for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void computeColor(float u, float v, unsigned char* color)
{
	float pi = 3.141592653f;

	float rad = sqrt(u * u + v * v);
	float a = atan2(-v, -u) / pi;
	float fk = (a + 1.0f) / 2.0f * (ncols-1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	float f = fk - k0;
	//f = 0; // uncomment to see original color wheel
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0f;
		float col1 = colorwheel[k1][b] / 255.0f;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		color[2 - b] = (int)(255.0f * col);
	}
}
