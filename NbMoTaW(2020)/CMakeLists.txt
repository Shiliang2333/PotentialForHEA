#
# Required preamble
#

cmake_minimum_required(VERSION 3.4)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})
find_package(KIM-API 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  enable_testing()
  project("${KIM_API_PROJECT_NAME}" VERSION "${KIM_API_VERSION}"
    LANGUAGES CXX C Fortran)
endif()

# End preamble


add_kim_api_model_library(
  NAME            "SNAP_LiChenZheng_2019_NbTaWMo__MO_560387080449_000"
  DRIVER_NAME     "SNAP__MD_536750310735_000"
  PARAMETER_FILES "NbMoTaW_Li_Arxiv2019.snapcoeff"
                  "NbMoTaW_Li_Arxiv2019.snapparam"
  )
