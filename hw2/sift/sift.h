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
class Descriptor;

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

	Mat build_interpolated_gaussian_table(unsigned size, double sigma);
	double gaussian2D(double x, double y, double sigma);


private:
	Mat *srcImage;
	int octaves;
	int intervals;
	int keypoint_num;

	Mat **gList;		//list of gaussian blurred images
	Mat **dogList;		//list of dog images
	Mat **extrema;		//list of extrema points
	double **absSigma;	//list of sigma used to blur image

	vector<Keypoint> keypoints;
	vector<Descriptor> descriptors;
};

class Keypoint{
public:
	Keypoint(){}
	Keypoint(double x,double y)
		:xi(x),yi(y){}
	Keypoint(double x, double y, vector<double> const& m, vector<double> const& o, unsigned s)
		:xi(x), yi(y), magnitudes(m), orientations(o), scale(s){}
public:
	double xi, yi;
	vector<double> magnitudes;
	vector<double> orientations;
	unsigned scale;
};

class Descriptor{
public:
	Descriptor(){}
	Descriptor(double x, double y, vector<double> const& f)
		:xi(x), yi(y), features(f){}
public:
	double xi, yi;
	vector<double> features;

};

#endif