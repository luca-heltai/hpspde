CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(gtest)

FIND_PACKAGE(deal.II 9.3 REQUIRED
HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Fix warning on mac
SET(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

ADD_COMPILE_DEFINITIONS(SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Gather files, libraries, and apps
file(GLOB LIB_SOURCES source/*.cc)
file(GLOB LIB_HEADERS include/*.h)
file(GLOB APP_SOURCES apps/*.cc)
FILE(GLOB GTEST_FILES gtests/*cc)

# Build both Release and Debug versions of library and executables
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")

      STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
  
      # Setup library
      SET(_lib "lab-lib${_p}")
      MESSAGE("-- Configuring library ${_lib}")
      
    IF(LIB_SOURCES)
      # Setup library
      SET(_lib "lab-lib${_p}")
      MESSAGE("-- Configuring library ${_lib}")
      ADD_LIBRARY(${_lib} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
      INSTALL(TARGETS ${_lib})
      SET(TEST_LIBRARIES_${_BUILD_TYPE} ${_lib})
    ENDIF(LIB_SOURCES)
      # Now setup all the executables
      foreach(sourcefile ${APP_SOURCES} )
          # I used a simple string replace, to cut off .cc.
          get_filename_component(name ${sourcefile} NAME_WE)
          
          # Add suffix
          set(name ${name}${_p})
          message("-- Configuring executable ${name}")
          add_executable( ${name} ${sourcefile} )
          # Make sure lab-lib is linked to each app
          if(LIB_SOURCES)
            target_link_libraries(${name} ${_lib})
          endif()  
          DEAL_II_SETUP_TARGET(${name} ${_BUILD_TYPE})
          INSTALL(TARGETS ${name})
      endforeach( sourcefile ${APP_SOURCES} )

      # Google tests testing
      FIND_PACKAGE(GTest)
      IF(GTest_FOUND)
          include(GoogleTest)
          set(lab_test "lab_test${_p}")
          ADD_EXECUTABLE(${lab_test} ${GTEST_FILES})
          if(LIB_SOURCES)
            target_link_libraries(${lab_test} ${_lib})
          endif()
          # Add the library to the test
          TARGET_LINK_LIBRARIES(${lab_test} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
          DEAL_II_SETUP_TARGET(${lab_test} ${_BUILD_TYPE})
          INSTALL(TARGETS ${lab_test})

          INCLUDE_DIRECTORIES(./include/ ${GTEST_INCLUDE_DIRS})

          ENABLE_TESTING()
          # GTEST_DISCOVER_TESTS(${lab_test})
          # GTEST_DISCOVER_TESTS(${fsi_book})
          MESSAGE("-- Configured Gtest executable ${lab_test}")
      ENDIF(GTest_FOUND)
    endif()
endforeach()