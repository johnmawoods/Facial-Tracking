#include <iostream>
#include <vector>

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>

#define NUM_OF_VERTICES	11510

using std::cout;
using std::endl;
using std::vector;

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

//vector<easy3d::vec3> getFaceVertsFromTensor(const arma::fcube& tensor, int id, int e) {
//
//	vector<easy3d::vec3> points;
//	points.reserve(NUM_OF_VERTICES);
//
//	for (int v = 0; v < NUM_OF_VERTICES; v++) {
//		int idx = 3 * v;
//		float x = tensor(idx, e, id);
//		float y = tensor(idx + 1, e, id);
//		float z = tensor(idx + 2, e, id);
//		points.push_back(easy3d::vec3(x, y, z));
//	}
//
//	return points;
//}

vector<easy3d::vec3> readFace3DFromObj(std::string path) {

	std::ifstream infile(path);
	if (infile.fail()) {
		std::cerr << "ERROR: couldn't open the Obj file to read the face from" << endl;
		exit(-1);
	}

	vector<easy3d::vec3> faceVerts;
	faceVerts.reserve(NUM_OF_VERTICES);

	for (int i = 0; i < NUM_OF_VERTICES; i++) {
		std::string hay;
		std::getline(infile, hay, ' ');
		std::getline(infile, hay, ' ');
		float x = std::stof(hay);
		std::getline(infile, hay, ' ');
		float y = std::stof(hay);
		std::getline(infile, hay);
		float z = std::stof(hay);

		faceVerts.push_back(easy3d::vec3(x, y, z));
	}

	infile.close();

	return faceVerts;
}


int main() {

	vector<uint32_t> meshIndices = readMeshTriangleIndicesFromFile("../data/GL/faces.obj");

	//int id = 102;
	//int e = 22;
	//vector<easy3d::vec3> faceVerts = getFaceVertsFromTensor(wholeTensor, id, e);

	vector<easy3d::vec3> faceVerts = readFace3DFromObj("../data/Tester_103/Blendshape/shape_22.obj");

	vector<int> vk = { 179, 214, 323, 501, 755, 765, 766, 767, 1642, 1717, 1902, 2122, 3185, 3226, 3239, 3272, 3434,
						3441, 3812, 3982, 4088, 4213, 4246, 4250, 4267, 4340, 5546, 6074, 6090, 6119, 6139, 6265, 6348,
						6350, 6502, 6576, 6703, 6744, 6826, 6870, 6880, 6986, 7079, 7122, 7140, 7161, 7165, 7238, 7256,
						7281, 7284, 7288, 7292, 7385, 8801, 8802, 8814, 8865, 8948, 8972, 8978, 9249, 10297, 10334,
						10453, 10536, 10629, 10682, 10684, 10760, 10820, 10844, 10892 };

	vector<easy3d::vec3> lmVerts(vk.size());
	for (int i = 0; i < vk.size(); i++)
		lmVerts[i] = faceVerts[vk[i]];

	//--- always initialize viewer first before doing anything else for 3d visualization 
	//========================================================================================
	easy3d::Viewer viewer("internal vertices");

	//------------------------- face surface mesh
	//===========================================================================
	auto surface = new easy3d::TrianglesDrawable("faces");
	surface->update_vertex_buffer(faceVerts);
	surface->update_element_buffer(meshIndices);
	surface->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

	//------------------- vertices corresponding to landmarks
	//===========================================================================
	auto vertices = new easy3d::PointsDrawable("vertices");
	vertices->update_vertex_buffer(lmVerts);
	vertices->set_uniform_coloring(easy3d::vec4(0.0, 0.9, 0.0, 1.0));
	vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
	vertices->set_point_size(10);

	//---------------------- add drawable objects to viewer
	//===========================================================================
	viewer.add_drawable(surface);
	viewer.add_drawable(vertices);

	viewer.fit_screen();
	viewer.run();

	return 0;
}