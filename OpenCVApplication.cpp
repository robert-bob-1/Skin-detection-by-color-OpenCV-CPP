// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Functions.h"
#include <fstream>
#include "opencv2/objdetect/objdetect.hpp"

wchar_t* projectPath;

CascadeClassifier face_cascade; // cascade clasifier object for face

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


////////////////////////////////////
// PROIECT DETECTAREA FETELOR PE CULOARE (UV CURS 7-8)
////////////////////////////////////

Mat getImage(string pathToImage) {
	Mat src = imread(pathToImage, IMREAD_COLOR);

	imshow("initial image", src);

	return src;
}

Mat getImageCIELUV(Mat src) {
	Mat dst = src.clone();

	cvtColor(src, dst, COLOR_BGR2Luv);
	//cvtColor(src, dst, COLOR_BGR2YCrCb);

	imshow("CIELUV image", dst);
	
	return dst;
}

Mat blurImage(Mat img) {
	Mat img_blurred;
	
	GaussianBlur(img, img_blurred, Size(5, 5), 0.8, 0.8);
	
	imshow("blurred image", img_blurred);
	
	return img_blurred;
}


#define HIST_SIZE 256

#define MAX_HUE 256
//variabile globale
int histG_hue[MAX_HUE]; // histograma globala / cumulativa

// variabile globale pentru proiect
int histU[HIST_SIZE]; // histograma culorii U
int histV[HIST_SIZE]; // histograma culorii V

// alte variabile pentru proiect
int maxHistVValue = 0;
int maxHistUValue = 0;

int debug = 1;

void computeMeanandStdDev(float& mean, float& std_dev, bool switchChannel) {

	int a, sat, i, j;
	int histF[HIST_SIZE]; // filtered histogram
	std::memset(histF, 0, HIST_SIZE * sizeof(unsigned int));

	//Filtrare histograma Hue (optional)
#define FILTER_HISTOGRAM 0
#if FILTER_HISTOGRAM == 1 //Filtering with a Gaussian filter of w=7
	float gauss[7];
	float sqrt2pi = sqrtf(2 * PI);
	float sigma = 1.5;
	float e = 2.718;
	float sum = 0;
	// Construire gaussian
	for (i = 0; i < 7; i++) { //compute Gaussian filter;
		gauss[i] = 1.0 / (sqrt2pi * sigma) * powf(e, -(float)(i - 3) * (i - 3)
			/ (2 * sigma * sigma));
		sum += gauss[i];
	}
	// Filtrare cu gaussian
	for (j = 3; j < HIST_SIZE - 3; j++)
	{
		for (i = 0; i < 7; i++)
			if (switchChannel)
				histF[j] += (float)histU[j + i - 3] * gauss[i];
			else
				histF[j] += (float)histV[j + i - 3] * gauss[i];
	}
#elif FILTER_HISTOGRAM == 0
	for (j = 0; j < MAX_HUE; j++)
		if (switchChannel)
			histF[j] = histU[j];
		else
			histF[j] = histV[j];
#endif // End of "Filtrare Gaussiana Histograma a"
	if (debug == 1)
		if (switchChannel)
			showHistogram("U histogram", histU, HIST_SIZE, 200, true);
		else
			showHistogram("V histogram", histV, HIST_SIZE, 200, true);

	int valMaxHist = 0;
	for (int i = 0; i < HIST_SIZE; ++i)
		if (switchChannel) {
			if (histU[i] > valMaxHist)
				valMaxHist = histU[i];
		}
		else {
			if (histV[i] > valMaxHist)
				valMaxHist = histV[i];
		}

	for (int i = 0; i < HIST_SIZE; ++i)
		if (histF[i] < valMaxHist / 10) //filter the values smaller than 10% of the maximum value
			histF[i] = 0;
	if (debug == 1)
		if (switchChannel)
			showHistogram("U filtered histogram", histF, HIST_SIZE, 200, true);
		else
			showHistogram("V filtered histogram", histF, HIST_SIZE, 200, true);


	int M = 0;
	int sum_hist = 0;
	float deviation = 0.0f;
	int L = HIST_SIZE;
	for (int i = 0; i < HIST_SIZE; ++i)
		M += histF[i];

	for (int histVal = 0; histVal < L; ++histVal)
		sum_hist += histVal * histF[histVal];

	float compute_mean = sum_hist * 1.0f / M; //mean

	for (int histVal = 0; histVal < L; ++histVal) {
		float diff = histVal * 1.0f - compute_mean;
		deviation += std::pow(diff, 2) * (histF[histVal] * 1.0f / M);
	}

	float compute_dev = std::sqrt(deviation);

	mean = compute_mean;
	std_dev = compute_dev;
	if (debug == 1)
		if (switchChannel) {
			cout << "Mean U: " << compute_mean << " StdDev U: " << compute_dev << endl;
		}
		else {
			cout << "Mean V: " << compute_mean << " StdDev V: " << compute_dev << endl;
		}
}

void train(float& meanU, float& meanV, float& stdDevU, float& stdDevV) {
	//Read images
	string pathToTrain = "Images/Proiect/Training/Sample";
	Mat images[12];
	for (int i = 1; i <= 12; i++) {
		string pathToImage = pathToTrain + to_string(i) + ".png";
		Mat RGBImage = getImage(pathToImage);
		Mat LuvImage = getImageCIELUV(RGBImage);

		images->push_back(LuvImage);
	}

	//Calculate histogram for U and V	
	memset(histU, 0, HIST_SIZE * sizeof(unsigned int));
	memset(histV, 0, HIST_SIZE * sizeof(unsigned int));
	

	for (int imgNo = 0; imgNo < 12; imgNo++) {
		for (int i = 0; i < images[imgNo].rows; i++) {
			for (int j = 0; j < images[imgNo].cols; j++) {
				//cout << to_string(images[imgNo].at<Vec3b>(i, j)[0]) << " " << to_string(images[imgNo].at<Vec3b>(i, j)[1]) << " " << to_string(images[imgNo].at<Vec3b>(i, j)[2]) << endl;
				int uValue = images[imgNo].at<Vec3b>(i, j)[1];
				int vValue = images[imgNo].at<Vec3b>(i, j)[2];
				histU[uValue]++;
				histV[vValue]++;
			}
		}
	}
	//if (debug == 1) {
	if (debug == 1) {
		showHistogram("Histogram U", histU, HIST_SIZE, HIST_SIZE);
		showHistogram("Histogram V", histV, HIST_SIZE, HIST_SIZE);
	}	

	//Compute mean and standard deviation for U and V
	computeMeanandStdDev(meanU, stdDevU, true);
	computeMeanandStdDev(meanV, stdDevV, false);
}

Mat computeLikelihood(Mat image, float meanU, float meanV, float stdDevU, float stdDevV) {
	Mat likelihood = Mat::zeros(image.rows, image.cols, CV_8UC1);
	float min = 100;
	float max = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int uValue = image.at<Vec3b>(i, j)[1];
			int vValue = image.at<Vec3b>(i, j)[2];
			float uComponent = pow((uValue - meanU + 127), 2);
			float vComponent = pow((vValue - meanV + 127), 2);
			float uDeviation = pow(stdDevU, 2);
			float vDeviation = pow(stdDevV, 2);
			float likelihoodInitial = sqrt(uComponent / uDeviation +
										   vComponent / vDeviation);
			likelihood.at<uchar>(i, j) = likelihoodInitial;

			if (likelihoodInitial > max)
				max = likelihoodInitial;
			if (likelihoodInitial < min)
				min = likelihoodInitial;
		}
	}
	cout << "Min: " << min << " Max: " << max << endl;
	if (debug == 1)
		imshow("Likelihood", likelihood);
	return likelihood;
}

Mat thresholding(Mat likelihoodImage, float meanU, float meanV, float stdDevU, float stdDevV) {
	int height = likelihoodImage.rows;
	int width = likelihoodImage.cols;
	Mat thresholdedImage = Mat(height, width, CV_8UC1);

	int histLikelihood[HIST_SIZE];
	memset(histLikelihood, 0, HIST_SIZE * sizeof(unsigned int));

	// Create histogram for likelihood image
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			histLikelihood[likelihoodImage.at<uchar>(i, j)]++;
	//if (debug == 1)
	showHistogram("Likelihood histogram", histLikelihood, HIST_SIZE, HIST_SIZE);


	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (likelihoodImage.at<uchar>(i, j) > 24)
				thresholdedImage.at<uchar>(i, j) = 255;
			else thresholdedImage.at<uchar>(i, j) = 0;
	
	imshow("Thresholded image", thresholdedImage);
	return thresholdedImage;
}

Mat cleanUpImage(Mat image) {
	Mat initialImage = image.clone();

	// Dilate and erode image
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	dilate(image, image, element);
	//imshow("dilated image", image);
	erode(image, image, element);
	//imshow("dilated and eroded image", image);

	// Fill closed in regions
	vector<vector<Point>> contours;

	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(image, contours, -1, Scalar(255), -1);

	//imshow("contour image", image);
	
	//erode and dilate again to remove leftover imperfections
	element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	erode(image, image, element);
	dilate(image, image, element);

	// And between images so that holes remain so we can filter out small objects with no holes
	bitwise_and(initialImage, image, image);
	if (debug == 1)
		imshow("before euler number cleanup", image);
	// Clean up objects using euler number (total number of objects in a region minus the total number of holes in those objects)
	// Only retain objects with euler number less than 1
	Mat outputImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
	Mat labels, stats, centroids;
	int numComponents = connectedComponentsWithStats(image, labels, stats, centroids);
	// Iterate through labeled objects and retain only those with euler number less than 1
	for (int i = 1; i < numComponents; ++i) {
		Mat mask = labels == i;
		Mat invertedMask;
		bitwise_not(mask, invertedMask);
		// Compute Euler's number
		Mat labelsForMask;
		int eulerNumber = connectedComponents(invertedMask, labelsForMask) - 1;
		int area = stats.at<int>(i, CC_STAT_AREA);
		cout << "Euler number: " << eulerNumber << " Area: " << area << endl;
		if (eulerNumber <= 1 && area < 500) {
			outputImage.setTo(0, mask);
		}
		else {
			outputImage.setTo(255, mask);
		}
	}

	imshow("cleaned image", outputImage);
	return outputImage;
}

Mat labelImage(Mat image) {
	// Find connected components using opencv's function
	Mat labels;
	int num_components = connectedComponents(image, labels);

	// Color objects
	Mat outputImage = Mat::zeros(image.rows, image.cols, CV_8UC3);
	RNG rng(0xFFFFFFFF); 

	for (int i = 1; i < num_components; i++) {
		Mat mask = labels == i;
		outputImage.setTo(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), mask);
	}

	imshow("labeled image", outputImage);
	return outputImage;
}

Rect FaceValidate(const string& window_name, Mat frame, int minFaceSize, bool hasFace) {

	std::vector<Rect> faces;
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
	equalizeHist(grayFrame, grayFrame);

	face_cascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	Rect faceROI = faces[0];
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[0], Scalar(0, 0, 255));

		Mat faceROI = grayFrame(faces[i]);
	}
	imshow(window_name, frame);
	return faceROI;
}

Mat FacesValidate(Mat inputImage) {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	
	Mat resultImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC3);
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return resultImage;
	}

	Mat inputImageGray;
	cvtColor(inputImage, inputImageGray, COLOR_BGR2GRAY);
	equalizeHist(inputImageGray, inputImageGray);

	imshow("before face detection", inputImageGray);
	vector<Rect> faces;
	face_cascade.detectMultiScale(inputImageGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10,10));

	for (const Rect& face : faces) {
		cout << "Face detected at: " << face.x << " " << face.y << endl;
		rectangle(resultImage, face, Scalar(255), 2);
	}

	imshow("faces", resultImage);

	return resultImage;

	/*Mat gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	char c;

	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {
		dst = Mat::zeros(gray.size(), gray.type());
		int minFaceSize = 500;
		Rect faceROI = FaceValidate("face", frame, minFaceSize, true);

		typedef struct {
			double arie;
			double xc;
			double yc;
		} mylist;
		vector<mylist> candidates;
		candidates.clear();
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3);
		findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		Moments m;
		if (contours.size() > 0)
		{
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				const vector<Point>& c = contours[idx];
				m = moments(c);
				double arie = m.m00;
				double xc = m.m10 / m.m00;
				double yc = m.m01 / m.m00;
				Scalar color(rand() & 255, rand() & 255, rand() & 255);
				drawContours(roi, contours, idx, color, FILLED, 8, hierarchy);
				mylist elem;
				elem.arie = arie;
				elem.xc = xc;
				elem.yc = yc;
				candidates.push_back(elem);
			}
		}
		if (candidates.size() >= 2)
		{
			mylist leftEye = candidates[0], rightEye = candidates[0];
			double arie1 = 0, arie2 = 0;
			for (mylist e : candidates)
			{
				if (e.arie > arie1)
				{
					arie2 = arie1;
					leftEye = rightEye;
					arie1 = e.arie;
					rightEye = e;
				}
				else
				{
					if (e.arie > arie2)
					{
						arie2 = e.arie;
						leftEye = e;
					}
				}
			}
			if ((abs(rightEye.yc - leftEye.yc) < 0.1 * faceROI.height && abs(rightEye.yc - leftEye.yc) < (faceROI.height) / 2))

				if (abs(rightEye.xc - leftEye.xc) > 0.3 * faceROI.width && abs(rightEye.xc - leftEye.xc) < 0.5 * faceROI.width)
					if (rightEye.xc - leftEye.xc > 0) {
						if (leftEye.xc <= (faceROI.width) / 2 && rightEye.xc >= (faceROI.width) / 2)
						{
							rectangle(frame, faceROI, Scalar(0, 255, 0));
							imshow("sursa", frame);
						}
					}
					else if (leftEye.xc >= (faceROI.width) / 2 && rightEye.xc <= (faceROI.width) / 2) {
						{
							DrawCross(roi, Point(leftEye.xc, leftEye.yc), 15, Scalar(0, 255, 255), 2);
							DrawCross(roi, Point(rightEye.xc, rightEye.yc), 15, Scalar(0, 255, 0), 2);
							rectangle(frame, faceROI, Scalar(0, 255, 0));
							imshow("sursa", frame);
						}
					}
		}
		imshow("colored", roi);
	}*/
}

double computeElongationFactor(const Mat& faceRegion) {
	// Compute covariance matrix
	Mat covarianceMatrix;
	calcCovarMatrix(faceRegion, covarianceMatrix, faceRegion, COVAR_NORMAL | COVAR_ROWS);

	// Compute eigenvalues and eigenvectors
	Mat eigenvalues, eigenvectors;
	eigen(covarianceMatrix, eigenvalues, eigenvectors);

	// Compute elongation factor
	double elongationFactor = sqrt(eigenvalues.at<double>(0, 0) / eigenvalues.at<double>(0, 1));

	return elongationFactor;
}

void drawPointInCenter(Mat& image, const Rect& boundingRect) {
	Point center(boundingRect.x + boundingRect.width / 2, boundingRect.y + boundingRect.height / 2);
	circle(image, center, 5, Scalar(255, 0, 0), -1);  // Draw a blue point at the center
}

void drawPointInCenter2(Mat& image, const Rect& boundingRect) {
	Point center(boundingRect.x + boundingRect.width / 2, boundingRect.y + boundingRect.height / 2);
	circle(image, center, 5, Scalar(255, 255, 0), -1);  // Draw a point at the center
}

Mat removeNotElongatedFaces(Mat image, Mat RGBImage, vector<double>& elongationFactors, vector<double>& areas) { // create using connectedComponents
	Mat dst = image.clone();

	Mat labels, stats, centroids;
	int numComponents = connectedComponentsWithStats(image, labels, stats, centroids);

	for (int i = 1; i < numComponents; ++i) {
		Mat mask = labels == i;
		
		// Calculate the bounding rectangle for the connected component
		Rect boundingRectObj = boundingRect(mask);

		// Compute the aspect ratio (elongation factor)
		double elongationFactor = static_cast<double>(boundingRectObj.height) / boundingRectObj.width;
		double area = stats.at<int>(i, CC_STAT_AREA);

		// Print or store the elongation factor for further analysis
		std::cout << "Elongation Factor for Component " << i << ": " << elongationFactor << std::endl;
		std::cout << "Area for Component " << i << ": " << area << std::endl;
		if (elongationFactor > 1 && elongationFactor < 3.5) {
			dst.setTo(255, mask);
			drawPointInCenter(RGBImage, boundingRectObj);
			areas.push_back(area);
			elongationFactors.push_back(elongationFactor);
		}
		else {
			dst.setTo(0, mask);
		}

		
	}
	cv::imshow("elongated faces", dst);

	cv::imshow("faces after elongation", RGBImage);

	return dst;
}

Mat prepareTemplateImage(Mat RGBTemplateImage) {
	// convert to grayscale, set to black all pixels with value higher than 200 (white-ish colors)
	Mat grayTemplateImage;
	cvtColor(RGBTemplateImage, grayTemplateImage, COLOR_BGR2GRAY);
	threshold(grayTemplateImage, grayTemplateImage, 200, 255, THRESH_BINARY_INV);
	imshow("template image", grayTemplateImage);

	return grayTemplateImage;
}

void templateMatching(Mat templateImage, Mat RGBImage, Mat image, vector<double> elongationFactors, vector<double> areas) {
	//Compute area and elongation of templateImage

	Mat templateFinalImage = image.clone();

	Mat labels, stats, centroids;
	int numComponents = connectedComponentsWithStats(image, labels, stats, centroids);

	for (int i = 1; i < numComponents; ++i) {
		Mat mask = labels == i;

		// Scale and rotate template image to match the current face and then compute crossmatch
		Rect boundingRectObj = boundingRect(mask);
		double elongationFactor = elongationFactors[i-1];
		double area = areas[i-1];

		Mat roi = image(boundingRectObj);

		// Calculate the angle using moments of the mask
		Moments mu = moments(mask, true);
		double angle = 0.5 * atan2(2 * mu.mu11, mu.mu20 - mu.mu02) * (180.0 / CV_PI);

		// Rotate template
		Point2f center(static_cast<float>(templateImage.cols / 2), static_cast<float>(templateImage.rows / 2));
		Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
		Mat rotatedTemplateImage;
		warpAffine(templateImage, rotatedTemplateImage, rotationMatrix, boundingRectObj.size());

		// Scale template

		// Compute crossmatch
		Mat result;
		matchTemplate(roi, rotatedTemplateImage, result, TM_CCORR_NORMED);

		if (debug == 1)
			imshow("crossmatch", result);

		// Find best match
		Point maxLoc;
		minMaxLoc(result, NULL, NULL, NULL, &maxLoc);

		// Draw Point in center of the face
		drawPointInCenter2(RGBImage, boundingRectObj);
	}

	cv::imshow("faces after template matching", RGBImage);
}


////////////////////////////////////
// De la Cristina Template Matching
////////////////////////////////////

void templateMatchingInRegion(const cv::Mat& source, const cv::Mat& templ, cv::Rect region, double threshold, std::vector<cv::Point>& matchLocations) {
	cout << "Dimensiunea regiunii" << region.size();
	cout << "Dimensiunea sablonului" << templ.size();

	if (region.width < templ.cols || region.height < templ.rows) {
		std::cerr << "Dimensiunea regiunii este mai mică decât dimensiunea șablonului." << std::endl;
		return;
	}

	cv::Mat sourceRegion = source(region);

	cout << "SourceRegion tip" << sourceRegion.type() << endl;
	cout << "Template type" << templ.type() << endl;
	if (sourceRegion.type() != templ.type()) {
		std::cerr << "Tipul de date al imaginii sursă și al șablonului trebuie să fie același." << std::endl;
		return;
	}

	cv::Mat result;
	cv::matchTemplate(sourceRegion, templ, result, cv::TM_CCOEFF_NORMED);

	cv::Point maxLoc;
	cv::minMaxLoc(result, nullptr, nullptr, nullptr, &maxLoc);

	if (result.at<float>(maxLoc.y, maxLoc.x) > threshold) {
		matchLocations.push_back(cv::Point(maxLoc.x + region.x + templ.cols / 2, maxLoc.y + region.y + templ.rows / 2));

		cv::rectangle(source, cv::Rect(maxLoc.x + region.x, maxLoc.y + region.y, templ.cols, templ.rows), cv::Scalar(0, 0, 255), 2);

		cv::Mat invertedTemplate;
		cv::bitwise_not(templ, invertedTemplate);

		cv::Mat templDraw = source.clone();
		invertedTemplate.copyTo(templDraw(cv::Rect(maxLoc.x + region.x, maxLoc.y + region.y, templ.cols, templ.rows)));
		cv::drawMarker(templDraw, cv::Point(maxLoc.x + region.x + templ.cols / 2, maxLoc.y + region.y + templ.rows / 2), cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 20, 2);
		imshow("Template Drawn", templDraw);
		waitKey(0);

	}
}

vector<Point> detectFacesAndOverlayTemplate(const cv::Mat& originalImage, const cv::Mat& templateImg, const cv::Mat& labels, int numRegions, double threshold) {

	cv::Mat imageWithOverlay = originalImage.clone();

	std::vector<cv::Point> matchLocations;

	for (int label = 1; label < numRegions; ++label) {
		cv::Mat currentRegionMask = (labels == label);

		float orientation;
		cv::Rect boundingRect;
		calculateRegionOrientationAndSize(currentRegionMask, orientation, boundingRect);
		cout << "Orientarea si dimensiunea regiunii sunt " << orientation << " " << boundingRect << endl;

		if (boundingRect.width > 0 && boundingRect.height > 0) {

			cv::Mat resizedTemplate;
			cv::resize(templateImg, resizedTemplate, boundingRect.size());

			imshow("tEMPLATE NOU", resizedTemplate);
			waitKey();

			if (boundingRect.width > 0 && boundingRect.height > 0) {

				templateMatchingInRegion(imageWithOverlay, resizedTemplate, boundingRect, threshold, matchLocations);


				waitKey(0);
			}
		}

	}

	return matchLocations;

}
void project()
{
	// Train model and compute histograms/means/deviations for both color channels
	float meanU, meanV, stdDevU, stdDevV;
	train(meanU, meanV, stdDevU, stdDevV);

	//Compute skin likelihood for images
	string pathToImage = "Images/Proiect/test.png";
	//string pathToImage = "Images/Proiect/averageFace.png";
	string pathToTemplateImage = "Images/Proiect/averageFace.png";
	//string pathToImage = "Images/kids.bmp";
	//string pathToImage = "Images/Persons/person_005.bmp";
	//string pathToImage = "Images/Lena_24bits.bmp";
	Mat RGBImage = getImage(pathToImage);
	Mat LuvImage = getImageCIELUV(RGBImage);
	Mat blurredImage = blurImage(LuvImage);
	Mat likelihoodImage = computeLikelihood(blurredImage, meanU, meanV, stdDevU, stdDevV);
	Mat thresholdedImage = thresholding(likelihoodImage, meanU, meanV, stdDevU, stdDevV);
	Mat cleanImage = cleanUpImage(thresholdedImage);
	vector<double> elongationFactors = vector<double>();
	vector<double> areas = vector<double>();
	Mat imageWithElongatedFaces = removeNotElongatedFaces(cleanImage, RGBImage, elongationFactors, areas);

	Mat RGBTemplateImage = getImage(pathToTemplateImage);
	Mat templateImage = prepareTemplateImage(RGBTemplateImage);
	//templateMatching(templateImage, RGBImage, imageWithElongatedFaces, elongationFactors, areas);
	//Mat labelledImage = labelImage(cleanImage);
	//Mat facesImage = FacesValidate(labelledImage);
	waitKey();
}



//////////////////////////////

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		op = 1;
		switch (op)
		{
			case 1:
				project();
				break;
		}
	} while (op != 0);
	return 0;
}
