//#define ENABLE_DEBUG 0
#if ENABLE_DEBUG
  #define logger std::cout << "LOG:" << __FILE__ << ":" << __LINE__ << " \t :: \t "
#else
  #define logger while(0) std::cout
#endif /* ENABLE_DEBUG */
