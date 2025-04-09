#include <gtest/gtest.h>

#include "interpolator.h"


using namespace dealii;

TEST(Interpolator, SmoothFunction)
{
  const std::string filename = SOURCE_DIR "/prms/01_smooth_interpolation.prm";
  InterpolatorParameters<2> parameters(filename);
  parameters.output_directory = SOURCE_DIR "/output";
  Interpolator<2> interpolator(parameters);
  interpolator.run();
  // Check the output files
  std::string output_file = parameters.output_directory + "/" +
                            parameters.output_file_name + "-2d-cycle-0.vtu";
  std::ifstream file(output_file);
  ASSERT_TRUE(file.good()) << "Output file " << output_file
                           << " does not exist or is not readable.";
  file.close();
  // Check the convergence table
  std::string convergence_file = parameters.output_directory + "/" +
                                 parameters.output_file_name +
                                 "-convergence.txt";
  std::ifstream conv_file(convergence_file);
  ASSERT_TRUE(conv_file.good()) << "Convergence file " << convergence_file
                                << " does not exist or is not readable.";
}