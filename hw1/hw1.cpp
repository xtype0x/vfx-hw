#include "opencv2/opencv.hpp"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
using namespace cv;
using namespace std;

#define LAMBDA 50
#define MAX_ITER 200000

//global variables
vector<Mat> images;
vector<float> times;
double g[256]={0.0};
double *exposure;

void loadExposureSeq();
double objectiveFunction(const gsl_vector *v, void *params);
void radianceMap();

int main(){
	loadExposureSeq();
	exposure = new double[images.size()];
	memset(exposure,0,sizeof(double)*images.size());
	radianceMap();


	delete [] exposure;

  	return 0;
}

void loadExposureSeq(){
	stringstream ss;
	float val[13] = {32,16,8,4,2,1,1.0/2,1.0/4,1.0/8,1.0/16,1.0/32,1.0/64,1.0/128};
	for (int i = 0; i < 13; ++i){
		//read the image
		string path;
		ss <<"test-case/img"<<setw(2)<<setfill('0')<<i+1<<".jpg";
		ss >> path;
		Mat img = imread(path);
		images.push_back(img);
		times.push_back(val[i]);
		ss.clear();
	}
}

double objectiveFunction(const gsl_vector *v, void *params){
	//initialize
	double g[256];
	double e[images.size()];
	double val = 0.0;
	int w;
	for(int i=0;i<256;i++)g[i] = gsl_vector_get (v, i);
	for(int i=0;i<images.size();i++)e[i]=gsl_vector_get (v,256+i);

	//calculate
	for (int i=0;i<images.size();++i)
		for(int j=0;j<images[i].rows;j++)
			for(int k=0;k<images[i].cols;k++){
				int z = (int)images[i].at<Vec3b>(j,k)[0];
				w = (z>127)?(255-z):z;
				val += pow(w*(g[z]-log(e[i])-log(times[i])),2);
			}
	double temp=0.0;
	for(int i=1;i<256-1;i++){
		w = (i>127)?(255-i):i;
		temp += pow((g[i-1]+g[i+1]-2*g[i])*w,2);
	}
	temp*=LAMBDA;
	val+=temp;

	return val;
}

void radianceMap(){
	//initialize
	size_t iter=0;
	int status;
	double size;
	int var_num = 256 + images.size();

	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer *s = NULL;
	gsl_vector *x, *ss;
  	gsl_multimin_function min_func;

  	min_func.n = var_num;
  	min_func.f = objectiveFunction;
  	min_func.params = NULL;
  	/*-- set starting point --*/
  	x = gsl_vector_alloc (var_num);
  	gsl_vector_set_all (x, 0.01);
  	/* Set initial step sizes to 1 */
  	ss = gsl_vector_alloc (var_num);
  	gsl_vector_set_all (ss, 10);

  	s = gsl_multimin_fminimizer_alloc (T, var_num);
  	gsl_multimin_fminimizer_set (s, &min_func, x, ss);

  	do{
		iter++;
    	status = gsl_multimin_fminimizer_iterate(s);
      
    	if (status) 
        	break;

      	size = gsl_multimin_fminimizer_size (s);
      	status = gsl_multimin_test_size (size, 1e-4);

      	if (status == GSL_SUCCESS){
			cout<<"converged to minimum at\n";
        }
        if(iter%100==1)
      	cout<<iter<<" val: "<<s->fval<<endl;
  	}while(status == GSL_CONTINUE && iter < MAX_ITER);
  	
  	for (int i = 0; i < 256; ++i){
  		g[i] = gsl_vector_get(x,i);
  	}
  	for(int i =0;i<images.size();i++){
  		exposure[i]=gsl_vector_get(x,256+i);
  	}
  	//free
  	gsl_vector_free(x);
  	gsl_vector_free(ss);
  	gsl_multimin_fminimizer_free (s);
}


