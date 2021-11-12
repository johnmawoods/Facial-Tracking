
#include "../include/tensor.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

// Reads vertices from each expression for each user in the warehouse
void buildRawTensor(string& warehousePath, string& outfile, tensor3& rawTensor) {
    warehousePath += "Tester_";

    // Each of the 150 users corresponds to one "shape.bs" file
    for (int i = 0; i < 150; i++) {
        string fileName = warehousePath + std::to_string(i + 1) + "/Blendshape/shape.bs";

        //string fileName = "D:/Desktop/2021FallSchool/CSE423/Github/Facial-Tracking/data/FaceWarehouse/Tester_" + std::to_string(i+1) + "/Blendshape/shape.bs";
        cout << "open file " << fileName << endl;
        
        FILE* fp;

        //cout << fileName << endl;
        fp = fopen(fileName.c_str(), "rb");
        int nShapes = 0, nVerts = 0, nFaces = 0;
        fread(&nShapes, sizeof(int), 1, fp);	  // nShape = 46
        //cout << nShapes << endl;
        fread( &nVerts, sizeof(int), 1, fp );	  // nVerts = 11510
        //cout << nVerts << endl;
        fread( &nFaces, sizeof(int), 1, fp );	  // nFaces = 11540
        //cout << nFaces << endl;


        for (int j = 0; j < 47; ++j)
            for (int k = 0; k < 11510; ++k)
                fread(&rawTensor(i, j, k), sizeof(Vector3f), 1, fp);

        fclose(fp);
    }

    writeTensor(outfile, rawTensor);
}

// Saves core tensor to binary file
void writeTensor(const string& filename, tensor3& tensor) {
    std::ofstream file(filename, std::ofstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 11510; k++)
                file.write(reinterpret_cast<const char*>(&tensor(i, j, k)), sizeof(Vector3f));
    file.close();
}

// Loads tensor from binary file
void loadRawTensor(const string& filename, tensor3& tensor) {
    std::ifstream file(filename, std::ifstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 11510; k++)
                file.read(reinterpret_cast<char *>(&tensor(i, j, k)), sizeof(Vector3f));

    file.close();
}

// Prints every vertex in the core tensor (81,145,500 vertices)
void displayEntireTensor(tensor3& tensor) {
    for(int i = 0; i < 150; i++) {
        for(int j = 0; j <47; j++) {
            for(int k = 0; k < 11510; k++) {
                cout << "User " << i << ", Expression " << j << ", Vertex " << k << ": " << tensor(i, j, k) << endl;
            }
        }
    }
}

void loadShapeTensor(string& SHAPE_TENSOR_PATH, tensor3& shapeTensor) {
    std::ifstream file(SHAPE_TENSOR_PATH, std::ifstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 73; k++)
                file.read(reinterpret_cast<char *>(&shapeTensor(i, j, k)), sizeof(Vector3f));

    file.close();
}

void buildShapeTensor(tensor3& rawTensor, string& outfile, tensor3& shapeTensor) {
    int shapeVerts[] = {10739, 3988, 3918, 3860, 10539, 489, 399, 3607, 9079, 9104, 1926, 6745, 2094, 2071, 9248, 9246, 7161, 7169, 1987, 7044, 6867, 697, 712, 10846, 3982, 10718, 4080, 3937, 192, 743, 6062, 10683, 3257, 4337, 10500, 4392, 8859, 4352, 10285, 10820, 3272, 750, 6057, 2103, 9296, 9224, 7276, 10407, 2176, 280, 9446, 7245, 7224, 4225, 9039, 10459, 3525, 6274, 6415, 9032, 6465, 6313, 7122, 8972, 3264, 3695, 3628, 1875, 8864, 6058, 1627, 6090, 6083};

    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 47; j++) {
            for (int k = 0; k < 73; k++) {
                shapeTensor(i, j, k) = rawTensor(i, j, shapeVerts[k]);
            }
        }
    }

    writeShapeTensor(outfile, shapeTensor);
}

void writeShapeTensor(const string& filename, tensor3& tensor) {
    std::ofstream file(filename, std::ofstream::binary);

    for (int i = 0; i < 150; i++)
        for (int j = 0; j < 47; j++)
            for (int k = 0; k < 73; k++)
                file.write(reinterpret_cast<const char*>(&tensor(i, j, k)), sizeof(Vector3f));
    file.close();
}


vector<cv::Point2f> readLandmarksFromFile_2(const std::string& path, const cv::Mat& image) {
    vector<int> orderedIndices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  //face contour
                                   21, 22, 23, 24, 25, 26,                            //left eyebrow
                                   18, 17, 16, 15, 20, 19,                            //right eyebrow
                                   27, 66, 28, 69, 29, 68, 30, 67,                    //left eye
                                   33, 70, 32, 73, 31, 72, 34, 71,                    //right eye
                                   35, 36, 37, 38, 44, 39, 45, 40, 41, 42, 43,        //nose contour
                                   65,												  //nose tip
                                   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,    //outer mouth
                                   63, 62, 61, 60, 59, 58						      //inner mouth
    };

    std::ifstream infile(path);
    if (infile.fail()) {
        std::cerr << "ERROR: unable to open the ladndmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
        exit(-1);
    }
    std::string hay;
    std::getline(infile, hay);
    int nOrigPoints = std::stof(hay);
    vector<float> origLms(nOrigPoints * 2);
    for (int i = 0; i < nOrigPoints; i++) {
        std::string temp;
        std::getline(infile, temp, ' ');
        origLms[i] = std::stof(temp) * image.cols;
        std::getline(infile, temp);
        origLms[i + nOrigPoints] = image.rows - (std::stof(temp) * image.rows);
    }
    infile.close();

    int nPoints = orderedIndices.size();
    vector<cv::Point2f> lms(nPoints);
    for (int i = 0; i < nPoints; i++) {
        lms[i].x = origLms[orderedIndices[i]];
        lms[i].y = origLms[orderedIndices[i] + nOrigPoints];
    }

    return lms;

}
