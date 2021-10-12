cmake_minimum_required(VERSION 3.0.0)
project(demo)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

#******************* definitions 
add_definitions(-D_HAS_STD_BYTE=0)             # for C++17
	
#******************* external packages 
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

#******************* executables 
file(GLOB HDR "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(demo
	${HDR}
	${SRC}
	${CMAKE_SOURCE_DIR}/example/main.cpp
)

#******************* linking
set_target_properties(demo PROPERTIES CXX_STANDARD 17)        # for C++17

target_link_libraries(demo
	${X11_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS}
)

#******************* DLL files
add_custom_command( TARGET demo POST_BUILD    # copy the dll files in the bin directory to the executable directory
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/bin" ${CMAKE_BINARY_DIR}  
)	
