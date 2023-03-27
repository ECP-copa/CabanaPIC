#ifndef diagnostics_h
#define diagnostics_h

#include <iostream>
#include <fstream>

template <class T>
struct PrintScreen
{
    void Print() {
	std::cout<<"Print screen\n";
    }
};

template <class T>
struct PrintFile
{
    std::ofstream myfile;
    PrintFile() : myfile("example.txt")
    {}
    void Print() {
	std::cout<<"Print to file\n";
	myfile << "Print to file\n";
    }
    ~PrintFile()
    {
	myfile.close();
    }
};

#endif

