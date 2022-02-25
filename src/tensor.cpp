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

        FILE* fp;
        fp = fopen(fileName.c_str(), "rb");

        int nShapes = 0, nVerts = 0, nFaces = 0;
        fread( &nShapes, sizeof(int), 1, fp );	  // nShape = 46
        fread( &nVerts, sizeof(int), 1, fp );	  // nVerts = 11510
        fread( &nFaces, sizeof(int), 1, fp );	  // nFaces = 11540

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
    int shapeVerts[] = { 5198, 1060, 4923, 1088, 5514, 10633, 10573, 9150, 1930, 1937, 8393, 9760, 9736, 7862, 5745, 3982, 4246, 4245, 10744, 608, 4088, 6954, 7138, 7142, 7145, 6981, 6967, 10892, 4383, 10905, 4420, 4401, 4370, 766, 767, 7293, 7272, 2188, 7234, 2193, 7284, 7281, 2180, 4226, 10418, 353, 3501, 3538, 6274, 9006, 1772, 1718, 1720, 2080, 8972, 3174, 3239, 3201, 6081, 8819, 1627, 6165, 6156, 8816, 6074, 3233, 3267, 227, 6168, 1652, 1648, 3272, 214 };
    int len = sizeof(shapeVerts) / sizeof(*shapeVerts);

    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 47; j++) {
            for (int k = 0; k < len; k++) {
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
        std::cerr << "ERROR: unable to open the landmarks file, refer to file " << __FILE__ << ", line " << __LINE__ << endl;
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

vector<uint32_t> readMeshTriangleIndicesFromFile(const std::string& path) {

    FILE* file = fopen(path.c_str(), "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        exit(-1);
    }

    vector<uint32_t> indices;
    indices.reserve(50000);

    while (true) {

        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        if (strcmp(lineHeader, "f") == 0) {

            unsigned int vertexIndex[4], uvIndex[4], normalIndex[4];

            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n",
                &vertexIndex[0], &uvIndex[0], &normalIndex[0],
                &vertexIndex[1], &uvIndex[1], &normalIndex[1],
                &vertexIndex[2], &uvIndex[2], &normalIndex[2],
                &vertexIndex[3], &uvIndex[3], &normalIndex[3]
            );

            for (int i = 0; i < 4; i++) {
                vertexIndex[i] -= 1;     // obj file indices start from 1
                uvIndex[i] -= 1;
                normalIndex[i] -= 1;
            }

            //====== change from quads to triangle
            indices.push_back(vertexIndex[0]);
            indices.push_back(vertexIndex[1]);
            indices.push_back(vertexIndex[2]);
            indices.push_back(vertexIndex[2]);
            indices.push_back(vertexIndex[3]);
            indices.push_back(vertexIndex[0]);


            if (matches != 12) {
                cout << "ERROR: couldn't read the faces! number of quads didn't match" << endl;
                exit(-1);
            }

        }

    }

    return indices;
}