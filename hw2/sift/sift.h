#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "sift.h"

using namespace cv;
using namespace std;

#ifndef SIFT_H
#define SIFT_H

class Sift;
class Keypoint;
class Desciprtor;

class Sift{
public:
	Sift(const char* );
	Sift();
	~Sift();
private:

};

class Keypoint{
public:
	Keypoint(){}
	Keypoint(float x,float y)
		:xi(x),yi(y){}
	Keypoint(float x, float y, vector<double> const& m, vector<double> const& o, unsigned s)
		:xi(x), yi(y), magnitudes(m), orientations(o), scale(s){}

	float xi, yi;
	vector<double> magnitudes;
	vector<double> orientations;
	unsigned scale;
};

class Desciprtor{
public:
	Desciprtor(){}
	Desciprtor(float x, float y, vector<double> const& f)
		:xi(x), yi(y), features(f){}

	float xi, yi;
	vector<double> features;

};

#endif