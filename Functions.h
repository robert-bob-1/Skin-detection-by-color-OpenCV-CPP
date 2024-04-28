#include "colorcode.h"
using namespace std;

#define COLOR_CONVENTION 1 // defined also in OpenLookView.cpp!
//define the color coding method: 
//	0 - HSI convention - http://www.had2know.com/technology/hsi-rgb-color-converter-equations.html 
//  1 - Middlebury convetion http://vision.middlebury.edu/flow/

#pragma once

void make_HSI2RGB_LUT();

Mat convert2flow(const Mat& velx, const Mat& vely);

//void calcOpticalFlowLK( const Mat& prev, const Mat& curr, Size winSize, Mat& flow );

//void calcOpticalFlowHS( const Mat& prev, const Mat& curr, int usePrevious, double lambda, TermCriteria criteria, Mat& flow);

void showFlow (const string& name, const Mat& gray, const Mat& flow, int mult=1, float minVel = 1.0f,
                            bool showImages = true, bool showVectors = false, bool showCircles = false);

void showFlowSparse (const string& name, const Mat& gray, const vector<Point2f>& prev_pts, const vector<Point2f>& curr_pts,
					const vector<uchar>& status, const vector<float>& error, int mult=1, 
					bool showImages = true, bool showVectors = false, bool showCircles = false);

void showFlowDense (const string& name, const Mat& gray, const Mat& flow, float minVel, bool showImages);

void showHistogram (const string& name, int *hist, const int  hist_cols, const int hist_rows, bool showImages);

void showHistogramDir (const string& name, int* hist, const int  hist_cols, const int hist_height, bool showImages);

Point RectCenter(Rect R);

int RectArea(Rect R);

void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness);

void Labeling(const string& name, const Mat& src, bool output_format);


