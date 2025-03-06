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

/*
<folder_path>/
	- frame_0.txt
	- frame_1.txt
	- frame_2.txt
	...

frame_0.txt:
\t|  center (x, y, z) | tracking_id | semantic label | index_pos |\n\t|-------------------|-------------|----------------|-----------|\n\t| x y z             | unknown     | pedestrian     | 0         |\n\t| x y z             | unknown     | vehicle.car    | 0         |\n\t| x y z             | unknown     | vehicle.car    | 1         |
*/


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

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}
inline std::string trim(std::string s) {
    rtrim(s);
    ltrim(s);
    return s;
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





struct ObjectData{
	double x, y, z;
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

    cout << "Loaded Frames: " << endl;
    for (const auto& frame : scene_frames){
        cout << frame << endl;
    }

    return 0;
}
