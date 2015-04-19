#ifndef SIFT_H
#define SIFT_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "sift.h"

using namespace cv;
using namespace std;

class Sift;
class Keypoint;
class Desciprtor;

class Sift{
public:
	Sift(const char* filename, unsigned o, unsigned i);
	Sift(Mat image, unsigned o, unsigned i);
	~Sift();

	void do_sift();

private:
	void generate_list();
	void build_scale_space();
	void detect_extrema();
	void assign_orientations();
	void extract_keypoint_descriptors();

	unsigned get_kernelsize(double sigma, double cut_off=0.001);
	Mat* build_interpolated_gaussian_table(unsigned size, double sigma);
	double gaussian2D(double x, double y, double sigma);


private:
	Mat *srcImage;
	unsigned octaves;
	unsigned intervals;
	int keypoint_num;

	Mat **gList;		//list of gaussian blurred images
	Mat **dogList;		//list of dog images
	Mat **extrema;		//list of extrema points
	double **absSigma;	//list of sigma used to blur image

	vector<Keypoint> keypoints;
	vector<Desciprtor> descriptors;
};

class Keypoint{
public:
	Keypoint(){}
	Keypoint(float x,float y)
		:xi(x),yi(y){}
	Keypoint(float x, float y, vector<double> const& m, vector<double> const& o, unsigned s)
		:xi(x), yi(y), magnitudes(m), orientations(o), scale(s){}
public:
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
public:
	float xi, yi;
	vector<double> features;

};

#endif