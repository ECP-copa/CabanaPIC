#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
//#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

TEST_CASE( "Trivial example", "[example_tests]" )
{
        REQUIRE(1);
}
