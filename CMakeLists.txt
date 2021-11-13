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
)

#******************* executables and linking


set(EASY3D_LIBRARIES easy3d::core easy3d::renderer easy3d::viewer easy3d::fileio)

set (COMMON_LIBRARIES 
	${X11_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS}
	${EASY3D_LIBRARIES}
)

file(GLOB HDR "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_library(asu
	${HDR}
	${SRC}
)

add_executable(3d_visual
	${CMAKE_SOURCE_DIR}/example/3d_visual.cpp
)
set_target_properties(3d_visual PROPERTIES CXX_STANDARD 17)        # for C++17
target_link_libraries(3d_visual
	${COMMON_LIBRARIES}
	asu
)

add_executable(pose_estimation
	${CMAKE_SOURCE_DIR}/example/pose_estimation.cpp
)
set_target_properties(pose_estimation PROPERTIES CXX_STANDARD 17)        # for C++17
target_link_libraries(pose_estimation
	${COMMON_LIBRARIES}
	asu
)

