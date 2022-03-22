cmake_minimum_required(VERSION 3.0.0)
project(demo)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

#******************* definitions 
add_definitions(-D_HAS_STD_BYTE=0)             # for C++17
	
#******************* external packages 
find_package(Easy3D REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(X11)

#******************* include directories 
set(EIGEN_INCLUDE_DIRS     ${CMAKE_SOURCE_DIR}/3rdparty/eigen)
set(ASU_INCLUDE_DIRS       ${CMAKE_SOURCE_DIR}/include)

include_directories(
	${EIGEN_INCLUDE_DIRS}	
	${OpenCV_INCLUDE_DIRS}	
	${ASU_INCLUDE_DIRS}	
	${CMAKE_SOURCE_DIR}/3rdparty/ceres/include
	${CMAKE_SOURCE_DIR}/3rdparty/glog/include
)

#******************* executables and linking
set(EASY3D_LIBRARIES easy3d::core easy3d::renderer easy3d::viewer easy3d::fileio)

set (COMMON_LIBRARIES 
	${X11_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS}
)

file(GLOB HDR "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(asu
	${HDR}
	${SRC}
)
target_link_libraries(asu
	${EASY3D_LIBRARIES}
)

#====================== project
add_executable(project
	${HDR}
	${SRC}
	${CMAKE_SOURCE_DIR}/src/main.cpp)
set_target_properties(project PROPERTIES CXX_STANDARD 17)        # for C++17
target_link_libraries(project
	${COMMON_LIBRARIES}
	${EASY3D_LIBRARIES}
	${CMAKE_SOURCE_DIR}/3rdparty/ceres/lib/ceres.lib
	${CMAKE_SOURCE_DIR}/3rdparty/glog/lib/glog.lib
	#asu
)

#====================== expression optimization
add_executable(optimization
	${CMAKE_SOURCE_DIR}/examples/optimization.cpp
)
set_target_properties(optimization PROPERTIES CXX_STANDARD 17)        # for C++17
target_link_libraries(optimization
	${COMMON_LIBRARIES}
	#asu
	${CMAKE_SOURCE_DIR}/3rdparty/ceres/lib/ceres.lib
	${CMAKE_SOURCE_DIR}/3rdparty/glog/lib/glog.lib
)
