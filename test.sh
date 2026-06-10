#!/bin/bash
gcc test.c -Wall -Wextra
echo "C test"
./a.out
g++ -x c++ test.c -Wall -Wextra
echo "C++ test"
./a.out
