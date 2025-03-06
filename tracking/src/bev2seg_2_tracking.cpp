// -- STL -- //
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <regex>

// -- Project -- //
#include <GTL/tracker_KF.h>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;
using namespace cv;

void print_help() {
    cout << "Usage: bev2seg_2_tracking <path_to_folder>\n\n";
    cout << "Este script toma la ruta a una carpeta que contiene los frames con objetos.\n";
    cout << "Cada archivo en la carpeta corresponde a un frame de la secuencia.\n";
    cout << "Cada archivo tiene información sobre los objetos presentes en ese frame.\n";
    cout << "File Format:\n";
    cout << "\t|  center (x, y, z) | tracking_id | semantic label | index_pos |\n\t|-------------------|-------------|----------------|-----------|\n\t| x y z             | unknown     | pedestrian     | 0         |\n\t| x y z             | unknown     | vehicle.car    | 0         |\n\t| x y z             | unknown     | vehicle.car    | 1         |\n";
    cout << "Example:\n";
    cout << "bev2seg_2_tracking ./frames/\n";
}

struct ObjectData{
	float x, y, z;
    string tracking_id;
    string semantic_label;
    int index_pos;
};

struct FrameData{
	int frame_num;
	vector<ObjectData> objects;
};

std::ostream& operator<<(std::ostream& os, const ObjectData& obj) {
    os << "x: " << obj.x << ", y: " << obj.y << ", z: " << obj.z
       << ", Tracking ID: " << obj.tracking_id
       << ", Semantic Label: " << obj.semantic_label
       << ", Index Pos: " << obj.index_pos;
    return os;
}
std::ostream& operator<<(std::ostream& os, const FrameData& frame) {
    os << "Frame Number: " << frame.frame_num << "\n";
    os << "Objects:\n";
    for (const auto& obj : frame.objects) {
        os << "  " << obj << "\n";  // Use the overloaded << for ObjectData
    }
    return os;
}


int extract_frame_number(const std::string& file_path) {
    std::string filename = fs::path(file_path).filename().string(); // "frame_19.txt"
    std::regex frame_regex(R"(frame_(\d+))");
    std::smatch match;
    if (std::regex_search(filename, match, frame_regex)) {
        return std::stoi(match[1].str()); 
    }
    return -1; // Return -1 if frame num not found
}

FrameData read_frame_data(string file_path){
	FrameData fdata;
    cout << "file_path:" << file_path << endl;

    int frame_num = extract_frame_number(file_path);
    fdata.frame_num = frame_num;
    if ( fdata.frame_num == -1 ) {
        return fdata;
    }

    // Read frame data
    ifstream file(file_path);
    if (! file.is_open()){
        cerr << "ERROR: Could not open file: " << file_path << endl;
        return fdata;
    }

    string line;
    while ( getline(file, line) ){
        istringstream iss(line);
        ObjectData obj;
        
        if (!(iss >> obj.x >> obj.y >> obj.z >> obj.tracking_id >> obj.semantic_label >> obj.index_pos)) {
            cerr << "ERROR: Could not read the line: " << line << endl;
            continue;
        }

        fdata.objects.push_back(obj);
    }
    file.close();
	return fdata;
}

int main (int argc, const char * argv[]) {
    cout << "BEV2Seg_2 Object tracker!\n";

    
    //****************************************************************************************
    // Load input data
    //****************************************************************************************
    // Check args
    if (argc != 2 || string(argv[1]) == "--help" || string(argv[1]) == "-h") {
        print_help();
        return 0;
    }
    string folder_path = argv[1];

    // Check path
    if (!( fs::exists(folder_path) & fs::is_directory(folder_path) )) {
        cerr << "Error: " << folder_path << " no es una ruta válida o no es una carpeta.\n";
        return 1;
    }

    // Read all the frames and get the objects data
    vector<FrameData> scene_frames;
	for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (fs::is_regular_file(entry.status())) {
            cout << "Procesando archivo: " << entry.path() << "\n";
            FrameData fdata = read_frame_data(entry.path());
			scene_frames.push_back(fdata);
            cout << endl;
		}
    }

	// Sort Frames
	std::sort(scene_frames.begin(), scene_frames.end(), [](const FrameData& fa, const FrameData& fb){
		return fa.frame_num < fb.frame_num; // Ascending order
	});


    //****************************************************************************************
    // Tracking initialization
    //****************************************************************************************
    std::unique_ptr<gtl::Tracker> tracker(new gtl::Tracker_KF()); 
    gtl::Tracker::Params params; 
    
    //****************************************************************************************
    // System and Measurement noise
    //****************************************************************************************
    int dims_ = 2;      // Variables for the state vector: x, y
    int derivs_ = 2;    // How many variables are derived: x, y
    
    // System and measurement noise for dynamic variables: x, y
    float system_noise_val_dev          = 0.1f;
    float system_noise_deriv_dev        = 0.1f;
	float groundTruth_meas_noise_dev    = 1.0f;

    // KF specific noise parameters
    // System noise for dynamic variables -> x, y, dx, dy
    std::vector<float> systemNoiseVar({
		system_noise_val_dev * system_noise_val_dev,        // x
		system_noise_val_dev * system_noise_val_dev,        // y
        system_noise_deriv_dev * system_noise_deriv_dev,    // dx
        system_noise_deriv_dev * system_noise_deriv_dev     // dy
    });
	
    // Measurement noise for dynamic variables -> x, y
	std::vector<float> measurementNoiseVar;
	for (int i = 0; i < dims_; i++) {
		measurementNoiseVar.push_back(groundTruth_meas_noise_dev * groundTruth_meas_noise_dev);
	}

    //********************************************************************************************
    // Association Function
    //********************************************************************************************
    // Distance between centroids
    std::vector<float> associationThresholds({ groundTruth_meas_noise_dev * 4.0f });
	std::unique_ptr<gtl::AssociationFunction> associationFunction(new gtl::AssociationFunctionGaussian(associationThresholds));
	
    //********************************************************************************************
    // State Vector
    //********************************************************************************************
    gtl::StateVector::StateVectorDimensions svd(dims_, 1, derivs_);

    // Fill-in parameters
	params.associationFunction = associationFunction.get();
	params.svDims = svd;	
	params.maxNumTracks = 5;        // Maximun number of current tracks
	params.num_min_instants_with_measurement = 2;       // if >= then birth
	params.num_max_instants_without_measurement = 4;    // if > then kill
	params.verbose = false;
	params.trackerParams[gtl::Tracker_KF::PARAM_SYSTEM_NOISE_VAR] = systemNoiseVar;
	params.trackerParams[gtl::Tracker_KF::PARAM_MEASUREMENT_NOISE_VAR] = measurementNoiseVar;
	tracker->setParams(params);


    //********************************************************************************************
    // Track data
    //********************************************************************************************

    for (const auto& frame : scene_frames){
        int fk = frame.frame_num;
        
        // For each frame get the observed data
        vector<gtl::Detection> detections_;
        for (const auto& obj : frame.objects){
            vector<float> measure_({obj.x, obj.y});
            string det_name = to_string(fk) + "_" + obj.semantic_label + "_" + to_string(obj.index_pos); 
            gtl::Detection detection_(gtl::StateVector(measure_), 1.0, obj.semantic_label, det_name);
            detections_.push_back(detection_);
        }

        // Track detections
        std::vector<gtl::Track> tracks = tracker->track(detections_, fk);


        // Visualization
        cv::Mat image(1080, 1080, CV_8UC3); // BGR
        cv::namedWindow("Tracking Debug", cv::WINDOW_NORMAL);
        
        for (const auto& det : detections_){
            vector<float> st = det.getStateVector().get();
            float x     = st[0];
            float y     = st[1];
            float dx    = st[2];
            float dy    = st[3];

            Point center(x, y);
            Scalar det_color(0, 255, 0); // Green
            cv::circle(image, center, 1.0, det_color, 2, CV_8U);
        }
        
        for (const auto& tr : tracks){
            tr.getFrameEnd();
            tr.getFrameStart();
            tr.getglobalID();
            ...
        }


        cv::imshow("Tracking Debug", image);
    }

    cv::destroyAllWindows();
    return 0;
}
