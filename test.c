#define NPYFILES_IMPLEMENTATION
#include "npyfiles.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float test_data[] = {
	1.0f, 2.0f, 3.0f,
	4.0f, 5.0f, 6.0f,
	7.0f, 8.0f, 9.0f,
};

#define check(c) if(!(c)) { fprintf(stderr, "check failed: %s\n", #c); exit(EXIT_FAILURE); } 

char buf[1<<20];

int main(void) 
{
	FILE *f = tmpfile();
	check(f);

	int64_t shape[] = {3,3};
	npy_write_file(f, test_data, sizeof(test_data), 2, shape, NPY_DTYPE_F32, 0);

	rewind(f);
	fread(buf, 1, 1<<20, f);

	NPYInfo info = npy_parse_file(buf, 0);

	check(info.dim==2);
	check(info.dtype==NPY_DTYPE_F32);
	check(info.shape[0]==3);
	check(info.shape[1]==3);
	check(info.is_native_byte_order);
	check(!memcmp(info.data_ptr, test_data, sizeof(test_data)));

	fclose(f);
	return 0;
}
