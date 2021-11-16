#include "utility.h"
#include <easy3d/core/surface_mesh.h>	
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>

#define NUM_OF_VERTICES	11510

vector<vector<uint32_t>> readQuadIndicesFromFile(const std::string& path) {

	FILE* file = fopen(path.c_str(), "r");
	if (file == NULL) {
		printf("Impossible to open the file !\n");
		exit(-1);
	}

	vector<vector<uint32_t>> quads;
	quads.reserve(50000);

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


			quads.push_back({ vertexIndex[0], vertexIndex[1], vertexIndex[2], vertexIndex[3] });


			if (matches != 12) {
				cout << "ERROR: couldn't read the faces! number of quads didn't match" << endl;
				exit(-1);
			}

		}

	}

	return quads;
}

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

	vector<vector<uint32_t>> quads = readQuadIndicesFromFile("../data/GL/faces.obj");
	vector<easy3d::vec3> faceVerts = readFace3DFromObj("../data/Tester_103/Blendshape/shape_22.obj");

	easy3d::Viewer viewer("surface mesh");

	easy3d::SurfaceMesh* mesh = new easy3d::SurfaceMesh();

	for (int i = 0; i < quads.size(); i++) {
		easy3d::SurfaceMesh::Vertex v0 = mesh->add_vertex(faceVerts[quads[i][0]]);
		easy3d::SurfaceMesh::Vertex v1 = mesh->add_vertex(faceVerts[quads[i][1]]);
		easy3d::SurfaceMesh::Vertex v2 = mesh->add_vertex(faceVerts[quads[i][2]]);
		easy3d::SurfaceMesh::Vertex v3 = mesh->add_vertex(faceVerts[quads[i][3]]);
		mesh->add_quad(v0, v1, v2, v3);
	}

	viewer.add_model(mesh);
	viewer.fit_screen();
	viewer.run();

	return 0;
}