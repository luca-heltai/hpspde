#include "interpolator.h"

int
main()
{
  try
    {
      InterpolatorParameters<2> parameters;
      Interpolator<2>           interpolator(parameters);
      interpolator.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }
  return 0;
}