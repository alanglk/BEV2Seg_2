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
    cout << "Usage: bev2seg_2_tracking <path_to_folder> <output_folder_path>\n\n";
    cout << "Este script toma la ruta a una carpeta que contiene los frames con objetos.\n";
    cout << "Cada archivo en la carpeta corresponde a un frame de la secuencia.\n";
    cout << "Cada archivo tiene información sobre los objetos presentes en ese frame.\n";
    cout << "File Format:\n";
    cout << "\n\tFrame_num: fk\n\tImage_path: <image-path>\n\tDetections --------------------\n\t|  detection (x, y, z)  | semantic_label |object_id        |\n\t|-----------------------|----------------|-----------------|\n\t| x y z                 | pedestrian     | 0_pedestrian_0  |\n\t| x y z                 | vehicle.car    | 0_vehicle.car_0 |\n\t| x y z                 | vehicle.car    | 0_vehicle.car_1 |\n\tTracks --------------------\n\t|  prediction (x, y, dx, dy) | frame_start | frame_end | tracking_id | semantic_label | associated detections  |\n\t|----------------------------|-------------|-----------|-------------|----------------|------------------------|\n\t| x y dx dy                  | 0           | 0         | -1          | pedestrian     | [ 0_pedestrian_0,...]  |\n\t| x y dx dy                  | 0           | 0         | -2          | vehicle.car    | [ 0_vehicle.car_0,...] |\n\t| x y dx dy                  | 0           | 0         | -3          | vehicle.car    | [ 0_vehicle.car_1,...] |\n";
    cout << "Example:\n";
    cout << "bev2seg_2_tracking ./frames/ ./output/\n";
}

/*
Frame_num: fk
Image_path: <image-path>
Detections --------------------
|  detection (x, y, z)  | semantic_label |object_id        |
|-----------------------|----------------|-----------------|
| x y z                 | pedestrian     | 0_pedestrian_0  |
| x y z                 | vehicle.car    | 0_vehicle.car_0 |
| x y z                 | vehicle.car    | 0_vehicle.car_1 |
Tracks --------------------
|  prediction (x, y, dx, dy) | frame_start | frame_end | tracking_id | semantic_label | associated detections  |
|----------------------------|-------------|-----------|-------------|----------------|------------------------|
| x y dx dy                  | 0           | 0         | -1          | pedestrian     | [ 0_pedestrian_0,...]  |
| x y dx dy                  | 0           | 0         | -2          | vehicle.car    | [ 0_vehicle.car_0,...] |
| x y dx dy                  | 0           | 0         | -3          | vehicle.car    | [ 0_vehicle.car_1,...] |
*/

struct ObjectData{
	float x, y, z;
    string semantic_label;
    string object_id; // framenum_semanticlabel_indexpos
};

struct FrameData{
	int frame_num;
    string image_path;
	vector<ObjectData> objects;
};

std::ostream& operator<<(std::ostream& os, const ObjectData& obj) {
    os << "x: " << obj.x << ", y: " << obj.y << ", z: " << obj.z
       << ", Semantic label: " << obj.semantic_label
       << ", Object ID: " << obj.object_id;
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

string vectorToString(const vector<string>& vec) {
    stringstream ss;
    ss << "[";
    
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << "\"" << vec[i] << "\"";
        if (i != vec.size() - 1) {
            ss << ", ";
        }
    }

    ss << "]";
    return ss.str();
}

FrameData read_frame_data(const string& file_path) {
    FrameData fdata;
    cout << "Reading file: " << file_path << endl;

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "ERROR: Could not open file: " << file_path << endl;
        return fdata;
    }

    string line;
    bool in_detections_section = false;

    while (getline(file, line)) {
        // Eliminar espacios en blanco al inicio y final
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (line.empty()) continue;  // Ignorar líneas vacías

        // Leer el frame number
        if (line.rfind("Frame_num:", 0) == 0) {  // Si la línea comienza con "Frame_num:"
            istringstream iss(line.substr(10));  // Extraer el número después de "Frame_num:"
            if (!(iss >> fdata.frame_num)) {
                cerr << "ERROR: Invalid frame number format in file: " << file_path << endl;
                return fdata;
            }
            continue;
        }

        // Leer el image path
        if (line.rfind("Image_path:", 0) == 0) {  // Si la línea comienza con "Image_path:"
            fdata.image_path = line.substr(11);  // Extrae todo el contenido después de "Image_path: "
            fdata.image_path.erase(0, fdata.image_path.find_first_not_of(" \t"));  // Elimina espacios extra
            continue;
        }

        // Detectar el inicio de la sección "Detections"
        if (line.find("Detections") != string::npos) {
            in_detections_section = true;
            continue;
        }

        // Detectar el inicio de la sección "Tracks" y salir de la lectura
        if (line.find("Tracks") != string::npos) {
            break;
        }

        // Leer los datos de detections
        if (in_detections_section) {
            istringstream iss(line);
            ObjectData obj;
            if (!(iss >> obj.x >> obj.y >> obj.z >> obj.semantic_label >> obj.object_id)) {
                cerr << "WARNING: Could not parse detection line: " << line << endl;
                continue;
            }
            fdata.objects.push_back(obj);
        }
    }

    file.close();
    return fdata;
}


void update_tracks_section(const string& file_path, const vector<gtl::Track>& tracks) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "ERROR: Could not open file for reading: " << file_path << endl;
        return;
    }

    string content;
    string line;
    // Leer el archivo y guardar todo el contenido
    while (getline(file, line)) {
        if (line.find("Tracks --------------------") != string::npos) {
            break;
        }
        content += line + "\n";
    }

    file.close();

    // Escribir la nueva sección de Tracks en el archivo
    ofstream fout(file_path, ios::out | ios::trunc);
    if (!fout.is_open()) {
        cerr << "ERROR: Could not open file for writing: " << file_path << endl;
        return;
    }

    fout << content; // Escribir el contenido original hasta la sección de tracks
    fout << "Tracks --------------------\n";

    // Escribir los nuevos tracks
    for (const auto& tr : tracks) {
        vector<float> st = tr.getStateVector().get();
        float x     = st[0];
        float y     = st[1];
        float z     = st[2];
        float dx    = st[3];
        float dy    = st[4];
        float dz    = st[5];

        int frame_start = tr.getFrameStart();
        int frame_end = tr.getFrameEnd();
        int track_id = tr.getglobalID();
        string semantic_label = tr.getClassName();
        string object_id = tr.getName();
        vector<string> associatedDetections;
        
        std::cout << "frame_start: " << frame_start << " frame_end: " << frame_end << std::endl;
        for (int i = frame_start; i <= frame_end; i++){
            if ( tr.hasDetection(i) ){
                gtl::Detection det = tr.getDetection(i);
                associatedDetections.push_back(det.getName());
                std::cout << "i: " << i << " Tracking id: " << track_id << " NumDetections: " << tr.getNumDetections() << " CountDet: " << tr.getCountDet() << " CountNoDet: " << tr.getCountNoDet() << " Name: " << det.getName() << std::endl;
            }else{
                std::cout << "i: " << i << " Tracking id: " << track_id << " NumDetections: " << tr.getNumDetections() << " CountDet: " << tr.getCountDet() << " CountNoDet: " << tr.getCountNoDet() << " Track doesnt have any detection for this frame" << std::endl;
            }
        }
        std::cout << std::endl;
        string ass_det_string = vectorToString(associatedDetections);
        
        // Escribir la nueva información de tracks
        fout << x << " " << y << " " << z << " " << dx << " " << dy << " " << dz << " " << frame_start << " " << frame_end << " " << track_id << " " << semantic_label << " " << ass_det_string << endl;
    }
    fout.close();
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
    fs::path input_folder_path = argv[1];

    // Check paths
    if (!( fs::exists(input_folder_path) & fs::is_directory(input_folder_path) )) {
        cerr << "Error: " << input_folder_path << " no es una ruta válida o no es una carpeta.\n";
        return 1;
    }
    // Read all the frames and get the objects data
    string file_base_name;
    vector<FrameData> scene_frames;
	for (const auto& entry : fs::directory_iterator(input_folder_path)) {
        if (fs::is_regular_file(entry.status())) {

            // Obtener el nombre base del archivo sin la extensión
            file_base_name = entry.path().stem().string();
            if (file_base_name.empty()) {
                cerr << "ERROR: Empty file base name" << endl;
                continue;
            }

            FrameData fdata = read_frame_data(entry.path());
			scene_frames.push_back(fdata);
		}
    }
    
    // File base name: "frame_"
    cout << "file_base_name: " << file_base_name << endl;
    size_t pos = file_base_name.find_last_of('_');
    if (pos != string::npos) {
        file_base_name = file_base_name.substr(0, pos);
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
    int dims_ = 3;      // Variables for the state vector: x, y, z
    int derivs_ = 3;    // How many variables are derived: x, y, z
    
    // System and measurement noise for dynamic variables: x, y, z
    float system_noise_val_dev          = 5.1f;
    float system_noise_deriv_dev        = 0.1f;
	float groundTruth_meas_noise_dev    = 4.0f;

    // KF specific noise parameters
    // System noise for dynamic variables -> x, y, dx, dy
    std::vector<float> systemNoiseVar({
		system_noise_val_dev * system_noise_val_dev,        // x
		system_noise_val_dev * system_noise_val_dev,        // y
		system_noise_val_dev * system_noise_val_dev,        // z
        system_noise_deriv_dev * system_noise_deriv_dev,    // dx
        system_noise_deriv_dev * system_noise_deriv_dev,    // dy
        system_noise_deriv_dev * system_noise_deriv_dev     // dz
    });
	
    // Measurement noise for dynamic variables -> x, y, z
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
	params.maxNumTracks = 10;        // Maximun number of current tracks
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
        std::cout << "Frame " << fk << " --------------------" << std::endl;
        
        // For each frame get the observed data
        std::cout << "Detections --------------------" << std::endl;
        vector<gtl::Detection> detections_;
        for (const auto& obj : frame.objects){
            vector<float> measure_({obj.x, obj.y, obj.z});
            std::cout << "object_id: " << obj.object_id << endl;
            gtl::Detection detection_(gtl::StateVector(measure_), 1.0, obj.semantic_label, obj.object_id);
            detections_.push_back(detection_);
        }

        // Track detections
        std::vector<gtl::Track> tracks = tracker->track(detections_, fk);


        // Save data
        std::cout << "Saving --------------------" << std::endl;
        string file_name = file_base_name + '_' + std::to_string(fk) + ".txt";
        fs::path file_path = input_folder_path / file_name;
        update_tracks_section(file_path, tracks);

    }
    cv::destroyAllWindows();
    return 0;
}
