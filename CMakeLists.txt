cmake_minimum_required(VERSION 2.8)
project(MySlam)

set(CMAKE_CXX_FLAGS "-std=c++11")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.1 REQUIRED)
find_package( CSparse REQUIRED )

include_directories(
	${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
    "/home/oleksandr/slambook/3rdparty/g2o/")

add_executable (MySlam src/RunSlam.cpp src/PnPProblem.cpp src/ModelRegistration.cpp src/CsvReader.cpp src/Mesh.cpp src/Utils.cpp src/RobustMatcher.cpp)

target_link_libraries(MySlam 
					  ${OpenCV_LIBS}
					  ${CSPARSE_LIBRARY}
   					  /home/oleksandr/slambook/3rdparty/g2o/lib/libg2o_core.so
   					  /home/oleksandr/slambook/3rdparty/g2o/lib/libg2o_stuff.so
   					  /home/oleksandr/slambook/3rdparty/g2o/lib/libg2o_types_sba.so
   					  /home/oleksandr/slambook/3rdparty/g2o/lib/libg2o_csparse_extension.so)
