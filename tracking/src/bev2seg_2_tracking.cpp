/*
* gtl - Generic Tracking Library
* gtl is part of viulib_dtc (Detection, Tracking and Classification) v13.10
*
* Copyright (C) Vicomtech (http://www.vicomtech.es/), 
* (Spain) all rights reserved.
* 
* viulib_dtc is a module of Viulib (Vision and Image Understanding Library) v13.10
* Copyright (C) Vicomtech (http://www.vicomtech.es/), 
* (Spain) all rights reserved.
* 
* As viulib_dtc depends on other libraries, the user must adhere to and keep in place any 
* licencing terms of those libraries:
*
* * OpenCV v3.0.0 (http://opencv.org/)
* 
* License Agreement for OpenCV
* -------------------------------------------------------------------------------------
* BSD 2-Clause License (http://opensource.org/licenses/bsd-license.php)
*
* The dependence of the "Non-free" module of OpenCV is excluded from viulib_dtc.
*
*/

// -- STL -- //
#include <random>
#include <numeric>

// -- Project -- //
#include <GTL/tracker_KF.h>
#include <opencv2/opencv.hpp>

using namespace std;

/** RECOMENDED KALMAN FILTER PARAMETERS:

	Measurement noise must always be as close as possible to the expected noise of the signal, with the same units.
	E.g. if the signal is expected to have 40.0f as std of noise, then, use 40.0f as measurement_noise_dev for KF.
	System noise depends on the expected miss-alignment between the signal behaviour, and the chosen linear model. The KF uses a linear
	function to express the evolution of the state of the signal (using derivatives: 1st order, 2nd order, etc.), therefore, the prediction will
	always be an approximation compared to the real signal. Even if the signal is synthetically created following the transition model, noise
	is always needed to allow the estimate to move away from the prediction. Otherwise, the system is stuck on the previous estimate.

	For 1st order model, the process noise for the derivative should be 0.1.

*/

enum signal_type { LINEAR, QUADRATIC, SINUSOIDAL };
signal_type type_experiment = LINEAR;
//cv::Vec2f runTracking(const std::vector<float>& _dataNoise, const std::vector<float>& _dataGT, float _measNoise, float _systemNoiseVal, float _systemNoiseDeriv)
//{
//	// Create a Kalman Filter tracker with optimal selection of measurement noise
//	int duration = static_cast<int>(_dataNoise.size());
//	std::unique_ptr<gtl::Tracker> tracker(new gtl::Tracker_KF());
//	gtl::Tracker::Params params;
//
//	// Define association function to assign detections to tracks (this only makes sense
//	// for multiple object tracking), but it is needed also for single-object
//	// Just need to define an association function which always assigns the new detection
//	// to the existing track: for that, let's use a very high euclidean distance
//	std::unique_ptr<gtl::AssociationFunction> associationFunction(new gtl::AssociationFunctionGaussian({_measNoise}));
//
//	// Define dimensions of the problem
//	gtl::StateVectorDimensions svd(1, 1, 1); // (x, dx)
//											 // Define noise of the problem (system and measurement)
//	vector<float> systemNoiseVar({ _systemNoiseVal*_systemNoiseVal, // x
//		_systemNoiseDeriv*_systemNoiseDeriv });	// dx		
//
//	vector<float> measurementNoiseVar({ _measNoise * _measNoise }); // x
//
//	// Fill-in parameters
//	params.associationFunction = associationFunction.get();
//	params.svDims = svd;	
//	params.maxNumTracks = 1;	
//	params.useRatio = false;
//	params.num_min_instants_with_measurement = 0;
//	params.num_max_instants_without_measurement = 0;
//	params.verbose = false;
//	params.trackerParams[gtl::Tracker_KF::PARAM_SYSTEM_NOISE_VAR] = systemNoiseVar;
//	params.trackerParams[gtl::Tracker_KF::PARAM_MEASUREMENT_NOISE_VAR] = measurementNoiseVar;
//	tracker->setParams(params);
//
//	// Track data
//	std::vector<float> data_tracked(duration, 0.0f);
//	std::vector<float> estimate_error(duration, 0.0f);
//	for (int t = 0; t < duration; t++)
//	{
//		// Run tracker
//		std::vector<float> measure({ _dataNoise[t] });
//		gtl::Detection detection(gtl::StateVector(measure), 1.0, "");
//		std::vector<gtl::Detection> detections({ detection });
//		std::vector<gtl::Track> tracks = tracker->track(detections, t);
//
//		// Read result		
//		gtl::StateVector sv = tracks[0].getStateVector();		
//		data_tracked[t] = sv.get()[0]; // x
//		float dx = sv.get()[1]; // dx, unused in this sample
//
//		estimate_error[t] = sv.get()[0] - _dataGT[t];
//	}
//
//	// Write error
//	float sum = std::accumulate(estimate_error.begin(), estimate_error.end(), 0.0f);
//	float mean = sum / estimate_error.size();
//
//	float sq_sum = std::inner_product(estimate_error.begin(), estimate_error.end(), estimate_error.begin(), 0.0f);
//	float stdev = std::sqrt(sq_sum / estimate_error.size() - mean * mean);
//
//	return cv::Vec2f(mean, stdev);
//}

void createLinearWithNoise(std::vector<float>& _data, std::vector<float>& _dataNoise, float _stddevNoise)
{
	// Prepare sine1 function
	int duration = 1000;
	_data = std::vector<float>(duration, 0.0);
	_dataNoise = std::vector<float>(duration, 0.0);

	float offset = 500.0f;
	float slope = 0.0f;//0.05f;

	// Prepare noise
	cv::RNG rng(cv::getCPUTickCount());	
	std::vector<float> noiseVector(duration, 0.0);
	float noiseMean = 0.0f;
	float noiseStddev = 0.0f;

	int countOutOf3Sigma = 0;

	for (int t = 0; t < duration; t++)
	{
		// Study statistics of the produced noise
		double noise = rng.gaussian(_stddevNoise);
		noiseVector[t] = static_cast<float>(noise);
		noiseMean += noiseVector[t];

		if (noise > 3 * _stddevNoise)
			countOutOf3Sigma++;
	}
	noiseMean /= duration;
	for (int t = 0; t < duration; t++)
	{
		noiseStddev += (noiseVector[t] - noiseMean)*(noiseVector[t] - noiseMean);
	}
	noiseStddev /= (duration - 1);
	noiseStddev = std::sqrt(noiseStddev);
	
	std::cout << "Noise signal produced: (mean, std)=(" << noiseMean << ", " << noiseStddev << "), with " << countOutOf3Sigma << " samples out of 3*sigma" << std::endl;
	
	for (int t = 0; t < duration; t++)
	{
		_data[t] = static_cast<float>(offset
			+ slope*t);
		_dataNoise[t] = static_cast<float>(_data[t] + noiseVector[t]);
	}
}
void createQuadraticWithNoise(std::vector<float>& _data, std::vector<float>& _dataNoise, float _stddevNoise)
{
	// Prepare sine1 function
	int duration = 1000;
	_data = std::vector<float>(duration, 0.0);
	_dataNoise = std::vector<float>(duration, 0.0);

	float offset = 500.0f;
	float slope = 0.001f;
	float slope2 = 0.002f;

	// Prepare noise
	cv::RNG rng(cv::getCPUTickCount());
	std::vector<float> noiseVector(duration, 0.0);
	float noiseMean = 0.0f;
	float noiseStddev = 0.0f;

	for (int t = 0; t < duration; t++)
	{
		// Study statistics of the produced noise
		double noise = rng.gaussian(_stddevNoise);
		noiseVector[t] = static_cast<float>(noise);
		noiseMean += noiseVector[t];
	}
	noiseMean /= duration;
	for (int t = 0; t < duration; t++)
	{
		noiseStddev += (noiseVector[t] - noiseMean)*(noiseVector[t] - noiseMean);
	}
	noiseStddev /= (duration - 1);
	noiseStddev = std::sqrt(noiseStddev);

	std::cout << "Noise signal produced: (mean, std)=(" << noiseMean << ", " << noiseStddev << ")" << std::endl;
	for (int t = 0; t < duration; t++)
	{
		_data[t] = static_cast<float>(offset
			+ slope * t
			+ slope2 *t*t);

		_dataNoise[t] = static_cast<float>(_data[t] + noiseVector[t]);
	}
}
void createSinusoidalWithNoise(std::vector<float>& _data, std::vector<float>& _dataNoise, float _stddevNoise)
{
	// Prepare sine1 function
	int duration = 1000;
	_data = std::vector<float>(duration, 0.0);
	_dataNoise = std::vector<float>(duration, 0.0);
	float offset = 500;
	float amplitude1 = 0;// 100; // 200
	float amplitude2 = 200;// 50;
	float amplitude3 = 100;// 100;
	float freq1 = 0.1f;//0.006f; // fast
	float freq2 = 0.01f;//0.06f; // slow
	float freq3 = 0.001f;//0.05f; // very slow

	// Prepare noise
	cv::RNG rng(cv::getCPUTickCount());
	std::vector<float> noiseVector(duration, 0.0);
	float noiseMean = 0.0f;
	float noiseStddev = 0.0f;

	for (int t = 0; t < duration; t++)
	{
		// Study statistics of the produced noise
		double noise = rng.gaussian(_stddevNoise);
		noiseVector[t] = static_cast<float>(noise);
		noiseMean += noiseVector[t];
	}
	noiseMean /= duration;
	for (int t = 0; t < duration; t++)
	{
		noiseStddev += (noiseVector[t] - noiseMean)*(noiseVector[t] - noiseMean);
	}
	noiseStddev /= (duration - 1);
	noiseStddev = std::sqrt(noiseStddev);

	std::cout << "Noise signal produced: (mean, std)=(" << noiseMean << ", " << noiseStddev << ")" << std::endl;

	for (int t = 0; t < duration; t++)
	{
		_data[t] = static_cast<float>(offset
			+ amplitude1 * std::sin(t*freq1)
			+ amplitude2 * std::sin(t*freq2)
			+ amplitude3 * std::sin(t*freq3));

		_dataNoise[t] = static_cast<float>(_data[t] + noiseVector[t]);
	}
}
void createSignal(std::vector<float>& _data, std::vector<float>& _dataNoise, float _stddevNoise, signal_type _type)
{
	if (_type == LINEAR)
		createLinearWithNoise(_data, _dataNoise, _stddevNoise);
	else if(_type == QUADRATIC)
		createQuadraticWithNoise(_data, _dataNoise, _stddevNoise);
	else if (_type == SINUSOIDAL)
		createSinusoidalWithNoise(_data, _dataNoise, _stddevNoise);
}

int main()
{
	// Define default parameters
	float system_noise_val_dev = 0.1f;
	float system_noise_deriv_dev = 0.1f;
	float groundTruth_meas_noise_dev = 40.0f;
	
	// Create data			
	std::vector<float> data, data_noise;
	createSignal(data, data_noise, groundTruth_meas_noise_dev, type_experiment);
	//createLinearWithNoise(data, data_noise, groundTruth_meas_noise_dev);
	//createSinusoidalWithNoise(data, data_noise, groundTruth_meas_noise_dev);
	int duration = static_cast<int>(data.size());

	// Create a Kalman Filter tracker with optimal selection of measurement noise
	std::unique_ptr<gtl::Tracker> tracker(new gtl::Tracker_KF());
	gtl::Tracker::Params params;

	// AssociationFunctionGaussian works by defining an association threshold for each observable dimension
	// The association between tracks and detections uses the provided number as a standard deviation to convert
	// euclidean distances into Mahalanobis distances. When the distance is > 1, that means the equivalent Euclidean
	// distance is higher than the provided number.
	// If this number is 3xnoise_stddev that means that a detection is declared as clutter wrt to a track if it falls
	// outside the range of 3*sigma of the gaussian defined around the predicted track, and thus directly comparable to the probability
	// defined by the density function (3*sigma = 99%, 2*sigma = 95%, 1*sigma = 68%).
	//
	// In the example below 4.0f * groundTruth_meas_noise_dev is used to ensure 99.99999% probability that noisy synthetic measurements
	// will be associated to the single existing track.
	// In other examples, 3*noise_stddev is possibly enough
	std::vector<float> associationThresholds({ groundTruth_meas_noise_dev*4.0f });
	std::unique_ptr<gtl::AssociationFunction> associationFunction(new gtl::AssociationFunctionGaussian(associationThresholds));
	
	// Define dimensions of the problem
	gtl::StateVector::StateVectorDimensions svd(1, 1, 1); // (x, dx)
											 // Define noise of the problem (system and measurement)
	vector<float> systemNoiseVar({ system_noise_val_dev*system_noise_val_dev, // x
		system_noise_deriv_dev*system_noise_val_dev });	// dx		

	vector<float> measurementNoiseVar({ groundTruth_meas_noise_dev * groundTruth_meas_noise_dev }); // x

	// Fill-in parameters
	params.associationFunction = associationFunction.get();
	params.svDims = svd;	
	params.maxNumTracks = 1;	
	params.num_min_instants_with_measurement = 0; // if >= then birth
	params.num_max_instants_without_measurement = 0; // if > then kill
	params.verbose = false;
	params.trackerParams[gtl::Tracker_KF::PARAM_SYSTEM_NOISE_VAR] = systemNoiseVar;
	params.trackerParams[gtl::Tracker_KF::PARAM_MEASUREMENT_NOISE_VAR] = measurementNoiseVar;

	tracker->setParams(params);

	// Track data
	std::vector<float> data_tracked(duration, 0.0f);
	std::vector<float> estimate_error(duration, 0.0f);
	for (int t = 0; t < duration; t++)
	{
		if(params.verbose)
			cout << "t: " << t << endl;
		// Run tracker
		std::vector<float> measure({ data_noise[t] });
		gtl::Detection detection(gtl::StateVector(measure), 1.0, "");
		std::vector<gtl::Detection> detections({ detection });
		std::vector<gtl::Track> tracks = tracker->track(detections, t);

		if (tracks.size() == 0)
			continue;

		// Read result		
		gtl::StateVector sv = tracks[0].getStateVector();
		
		data_tracked[t] = sv.get()[0]; // x
		float dx = sv.get()[1]; // dx, unused in this sample

		estimate_error[t] = sv.get()[0] - data[t];
		
	}

	// Draw data
	cv::Mat image(1000, 1000, CV_8UC3);
	image.setTo(0);
	for (int t = 1; t < duration; t++)
	{
		// Draw in grey the association threshold: outside this limit, the Mahalanobis distance is higher than 1, and then association fails (clutter is activated)
		cv::line(image, cv::Point(t, cvRound(data_tracked[t - 1] - associationThresholds[0])), cv::Point(t, cvRound(data_tracked[t - 1] + associationThresholds[0])), cv::Scalar(127, 127, 127), 2, CV_8U);

		cv::line(image, cv::Point(t, cvRound(data[t])), cv::Point(t - 1, cvRound(data[t - 1])), cv::Scalar(0, 255, 0), 2, CV_8U);
		cv::line(image, cv::Point(t, cvRound(data_noise[t])), cv::Point(t - 1, cvRound(data_noise[t - 1])), cv::Scalar(0, 0, 255), 1, CV_8U);
		cv::line(image, cv::Point(t, cvRound(data_tracked[t])), cv::Point(t - 1, cvRound(data_tracked[t - 1])), cv::Scalar(255, 0, 0), 1, CV_8U);

		//cv::line(image, cv::Point(t, cvRound(abs(estimate_error[t]))), cv::Point(t - 1, cvRound(abs(estimate_error[t - 1]))), cv::Scalar(127, 127, 127), 1, CV_8U);		
		cv::imshow("image", image);
		cv::waitKey(10);
	}
	//cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", image);
	cv::waitKey(0);


}