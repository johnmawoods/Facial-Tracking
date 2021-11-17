#include "utility.h"

#include "3rd_party/glfw/include/GLFW/glfw3.h"	// for the KEYs

namespace asu {
	class CustomViewer : public easy3d::Viewer {
	public:
		CustomViewer(const std::string& title) : easy3d::Viewer(title) {}

	protected:
		bool key_press_event(int key, int modifiers) override {
			if (key == GLFW_KEY_ESCAPE && modifiers == 0)
				this->exit();
			else
				return easy3d::Viewer::key_press_event(key, modifiers);
		}
	};
}

int main() {

	asu::Utility util;
	vector<vector<uint32_t>> quads = util.readQuadIndicesFromFile("../data/GL/faces.obj");
	vector<easy3d::vec3> faceVerts = util.readFace3DFromObj("../data/Tester_103/Blendshape/shape_22.obj");
	//vector<easy3d::vec3> faceVerts = util.readFace3DFromObj("../data/pose_16.obj");

	//-------------------------------- viewer
	//===================================================================================
	asu::CustomViewer viewer("custom viewer");   // added the esc button for exiting
	easy3d::Camera* cam = viewer.camera();       // visualized axes at the bottom left of the screen are red(x), green(y), blue(z) 
	cam->setUpVector(easy3d::vec3(0, 1, 0));
	cam->setViewDirection(easy3d::vec3(0, 0, -1));

	//-------------------------------- point cloud
	//===================================================================================
	easy3d::PointCloud* cloud = new easy3d::PointCloud;
	for (float i = -2; i <= 2; ++i) {
		for (float j = -2; j <= 2; ++j)
			cloud->add_vertex(easy3d::vec3(i, j, 0));  // z = 0: all points are on XY plane.
	}
	auto colors = cloud->add_vertex_property<easy3d::vec3>("v:color");    // per vertex properties are: color, normal and point
	for (auto v : cloud->vertices())
		colors[v] = easy3d::random_color();		        // assign a random color to point 'v'

	viewer.add_model(cloud);             // renderer and manipulator are initialized after adding the model, so use them after this line
	auto vDrawable = cloud->renderer()->get_points_drawable("vertices");   // the string must be "vertices"
	//vDrawable->set_uniform_coloring(easy3d::vec4(0.8, 0, 0.0, 1.0));
	vDrawable->set_impostor_type(easy3d::PointsDrawable::SPHERE);
	vDrawable->set_point_size(20);
	float pi = 3.1415926;
	cloud->manipulator()->frame()->rotate(easy3d::quat(easy3d::vec3(0, 0, 1), pi / 4.0));    // 45 degrees around z
	cloud->manipulator()->frame()->translate(easy3d::vec3(0, 0, 1));

	//-------------------------------- surface mesh
	//===================================================================================
	easy3d::SurfaceMesh* mesh = new easy3d::SurfaceMesh();
	for (int i = 0; i < quads.size(); i++) {
		easy3d::SurfaceMesh::Vertex v0 = mesh->add_vertex(faceVerts[quads[i][0]]);
		easy3d::SurfaceMesh::Vertex v1 = mesh->add_vertex(faceVerts[quads[i][1]]);
		easy3d::SurfaceMesh::Vertex v2 = mesh->add_vertex(faceVerts[quads[i][2]]);
		easy3d::SurfaceMesh::Vertex v3 = mesh->add_vertex(faceVerts[quads[i][3]]);
		mesh->add_quad(v0, v1, v2, v3);
	}
	
	viewer.add_model(mesh);        // the model must first be added to the viewer before accessing the drawables
	auto sDrawable = mesh->renderer()->get_triangles_drawable("faces");    // the string must be "faces"
	sDrawable->set_smooth_shading(true);
	sDrawable->set_uniform_coloring(easy3d::vec4(0.8, 0.8, 0.8, 1.0));

	//-------------------------------- rendering
	//===================================================================================
	viewer.fit_screen();
	viewer.run();

	return 0;
}