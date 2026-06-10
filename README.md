# npyfiles.h

This is a public domain single-header C library for reading (a subset of) .npy files.
It only supports numeric arrays, no strings or arbitrary Python objects, but IME that's
the most common use case for Python-C data interop. 
It is a single-header library in the style of Sean Barrett (https://github.com/nothings/):
The file `npyfiles.h` contains both the header and the library implementation.
To use this library, copy `npyfiles.h` into your project, and define 
`NPYFILES_IMPLEMENTATION` in exactly one `.c` file, before you include it.

Compiles in C++.
