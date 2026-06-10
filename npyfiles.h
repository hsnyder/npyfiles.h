/*
	Harris M. Snyder, 2023
	This is free and unencumbered software released into the public domain.

	npyfiles.h: a library for .npy file IO from C or C++.
	It only supports a subset (numeric arrays, no strings or arbitrary Python
	objects), but IME that's the most common use case for Python-C data interop. 

	This is a "stb-style" single-header library.  Define NPYFILES_IMPLEMENTATION 
	in one translation unit before you include npyfiles.h. If you haven't seen 
	this trick before, see github.com/nothings/stb.

	USAGE

	  I recommend just reading the header portion of this file, there are inline
	  comments describing each function. If you're in a hurry, you probably want:
	  
	    npy_parse_file

	    Parses the header of an already ingested (or mmap'd) .npy file, and
	    returns a struct containing dtype, shape, etc, and a pointer to where
	    the actual array data starts. You can then copy out that data or use
	    it directly if it's appropriately aligned.  
	 
	  and 
	    
	    npy_write_file

	    Given a data pointer, shape and dtype info, etc, as arguments, writes an
	    encoded .npy file to a given FILE * stream. Unfortunately there's no
	    version to encode directly into a memory buffer right now. 


	ERROR HANDLING

	  const char **errmsg

	  Several functions have an `errmsg` argument. If that argument is null,
	  then errors within the function are fatal (print to stderr and call exit).
	  If it IS provided, then an error causes it to be set to a STATIC string
	  describing the problem (don't try to free it).
	  
*/




#ifndef NPYFILES_H
#define NPYFILES_H

#ifndef NPY_API
#define NPY_API
#endif

#include <stdint.h>
#include <stdio.h>

typedef enum {
	NPY_DTYPE_I8,
	NPY_DTYPE_I16,
	NPY_DTYPE_I32,
	NPY_DTYPE_I64,
	NPY_DTYPE_U8,
	NPY_DTYPE_U16,
	NPY_DTYPE_U32,
	NPY_DTYPE_U64,
	NPY_DTYPE_F16,
	NPY_DTYPE_F32,
	NPY_DTYPE_F64,
	NPY_DTYPE_C64,    // complex: pair of f32
	NPY_DTYPE_C128,   // comples: pair of f64
} NPYDataType;

typedef struct {
	int         dim;
	NPYDataType dtype;
	int64_t     shape[6];
	int	    is_native_byte_order;
	void        *data_ptr;
} NPYInfo;

static int64_t npy_nel(NPYInfo info)
{
	int64_t n = info.dim > 0 ? 1 : 0;
	for(int d=0; d < info.dim; d++) {
		n *= info.shape[d];
	}
	return n;
}

static int npy_dtype_size(int dtype)
{
	switch(dtype) {
	case NPY_DTYPE_I8:  case NPY_DTYPE_U8:  return 1;
	case NPY_DTYPE_I16: case NPY_DTYPE_U16: case NPY_DTYPE_F16: return 2;
	case NPY_DTYPE_I32: case NPY_DTYPE_U32: case NPY_DTYPE_F32: return 4;
	case NPY_DTYPE_I64: case NPY_DTYPE_U64: case NPY_DTYPE_F64: return 8;
	case NPY_DTYPE_C64:  return 8;
	case NPY_DTYPE_C128: return 16;
	default: return 0;
	}
}


NPY_API NPYInfo
npy_parse_file(void *file_buf, const char **errmsg);
// Assumes you've read the whole file into memory already


// TODO we need a version of this that just encodes to memory
NPY_API void
npy_write_file(FILE *f, void* data, int64_t nbytes, int dim, int64_t *shape /*[dim]*/, int dtype, const char **errmsg);
// Does not close `f` after completing.


NPY_API void
npy_copy_data(NPYInfo info, void *dst, int64_t dst_bytes, const char **errmsg);
// Copy the tensor data referenced by `info.data_ptr` into `dst`,
// converting from the file's byte order to host byte order on the
// way out. `dst` must be at least npy_nel(info) * npy_dtype_size(info.dtype)
// bytes (pass that as `dst_bytes` for the size check). The source
// is NOT mutated, so you can use it with mmap'd files if you wish. 
// If the file is already in native byte order, this is just a memcpy.


#endif



#ifdef NPYFILES_IMPLEMENTATION

#include <inttypes.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
static void npy_errmsg(const char **errmsg_ptr, const char *errmsg)
{
	if(errmsg_ptr) {
		*errmsg_ptr = errmsg;
	} else {
		fprintf(stderr, "%s\n", errmsg);
		exit(EXIT_FAILURE);
	}
}

static int npy_host_is_little_endian(void)
{
    union {
        unsigned int i;
        unsigned char c[sizeof(unsigned int)];
    } test;
	test.i = 1;
    return test.c[0] == 1;
}



static int64_t 
npy_parse_positive_int(char **ptr, char *limit)
{
	/*
		- Skips preceding spaces and tabs
		- Won't read past 'limit'
		- Doesn't check for integer overflow
		- Doesn't parse negative numbers
		- Returns -1 on failure
	*/
	char * str = *ptr;

	while(*str == ' ' || *str == '\t') str++;

	int64_t v = 0;
	int n = 0;
	while (str < limit  &&  *str >= 48  &&  *str <= 57)
	{
		int digit = *str - 48;
		v *= 10;
		v += digit;
		str++;
		n++;
	}

	*ptr = str;
	return n > 0 ? v : -1;
}

static int
npy_consume (char **ptr, char *limit, const char *expected)
{
	while(**ptr == ' ' || **ptr == '\t') (*ptr) += 1;
	int len = strlen(expected);
	if (*ptr + len >= limit) return 0;
	if (memcmp(*ptr, expected, len)) return 0;
	(*ptr) += len;
	return 1;
}

static int
npy_optional (char **ptr, char *limit, const char *expected)
{
	char *backup = *ptr;
	int yes = npy_consume(ptr,limit,expected);
	if(!yes) *ptr = backup;
	return yes;
}

static int 
npy_round_up (int value, int to)
{
	int mod = value%to;	
	if(mod) return value+to-(value%to);
	return value;
}

#define npy_ERRMSG(errmsg_ptr, errmsg_string) \
	do { \
		npy_errmsg((errmsg_ptr), (errmsg_string)); \
		NPYInfo nullreturn; \
		memset(&nullreturn, 0, sizeof(nullreturn)); \
		return nullreturn; \
	} while(0)

NPY_API NPYInfo
npy_parse_file(void *file_buf, const char **errmsg)
// Assumes you've read the whole file into memory already
{
	NPYInfo info;
	memset(&info, 0, sizeof(info));

	int host_is_le = npy_host_is_little_endian();

	uint8_t * header = (uint8_t*)file_buf;

	if (memcmp(header, "\x93NUMPY", 6)) npy_ERRMSG(errmsg, "Invalid magic string");
	header += 6;

	uint8_t major_ver = *header++;
	uint8_t minor_ver = *header++;

	if (minor_ver != 0) npy_ERRMSG(errmsg, "Unrecognized .npy version (minor version)");
	if (major_ver != 1 && major_ver != 2 && major_ver != 3) npy_ERRMSG(errmsg, "Unrecognized .npy version (major version)");

	uint32_t header_len = 0;

	header_len = *header++;
	header_len |= (uint32_t)(*header++) << 8;

	if (major_ver > 1) {
		header_len |= (uint32_t)(*header++) << 16;
		header_len |= (uint32_t)(*header++) << 24;
	}

	char byte_order_mark = ' ';

	info.data_ptr = header + header_len;

	char *t = (char*)header;
	char *e = t + header_len;

	if (!npy_consume(&t,e,"{")) npy_ERRMSG(errmsg, "Expected '{'");

	for(int i = 0; i < 3; i++) {
		if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
			npy_ERRMSG(errmsg, "Expected dict key string");

		if (npy_optional(&t,e,"descr")) {
			if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
				npy_ERRMSG(errmsg, "Unrecognized or unterminated dict key");

			if (!npy_consume(&t,e,":"))
				npy_ERRMSG(errmsg, "Expected ':' following key 'descr'");

			if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
				npy_ERRMSG(errmsg, "Couldn't parse value for 'descr' (unsupported type?)");

			if (npy_optional(&t,e,"<")) {
				// data is LE
				if(host_is_le) info.is_native_byte_order = 1;
				else           info.is_native_byte_order = 0;
				byte_order_mark = '<';
			} else if (npy_optional(&t,e,">")) {
				// data is BE
				if(host_is_le) info.is_native_byte_order = 0;
				else           info.is_native_byte_order = 1;
				byte_order_mark = '>';
			} else if (npy_consume(&t,e,"|")) {
				info.is_native_byte_order = 1;
				byte_order_mark = '|';
			} else {
				npy_ERRMSG(errmsg, "Unsupported byte order marker");
			}

			if (npy_consume(&t,e,"f")) {
				if(byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid floating-point data type (byte order mark)");
				switch(npy_parse_positive_int(&t, e)) {
					case 2: info.dtype = NPY_DTYPE_F16; break;
					case 4: info.dtype = NPY_DTYPE_F32; break;
					case 8: info.dtype = NPY_DTYPE_F64; break;
					default: npy_ERRMSG(errmsg, "Unsupported floating-point data type");
				}
			} else if (npy_consume(&t,e,"c")) {
				if(byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid complex data type (byte order mark)");
				switch(npy_parse_positive_int(&t, e)) {
					case 8:  info.dtype = NPY_DTYPE_C64;  break;
					case 16: info.dtype = NPY_DTYPE_C128; break;
					default: npy_ERRMSG(errmsg, "Unsupported complex data type");
				}
			} else if (npy_consume(&t,e,"i1")) {
				if(byte_order_mark != '|' && byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid i1 data type (byte order mark)");
				info.dtype = NPY_DTYPE_I8;
			} else if (npy_consume(&t,e,"u1")) {
				if(byte_order_mark != '|' && byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid u1 data type (byte order mark)");
				info.dtype = NPY_DTYPE_U8;
			} else if (npy_consume(&t,e,"i")) {
				if(byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid signed integer data type (byte order mark)");
				switch(npy_parse_positive_int(&t, e)) {
					case 2: info.dtype = NPY_DTYPE_I16; break;
					case 4: info.dtype = NPY_DTYPE_I32; break;
					case 8: info.dtype = NPY_DTYPE_I64; break;
					default: npy_ERRMSG(errmsg, "Unsupported signed integer data type");
				}
			}  else if (npy_consume(&t,e,"u")) {
				if(byte_order_mark != '<' && byte_order_mark != '>')
					npy_ERRMSG(errmsg, "Invalid unsigned integer data type (byte order mark)");
				switch(npy_parse_positive_int(&t, e)) {
					case 2: info.dtype = NPY_DTYPE_U16; break;
					case 4: info.dtype = NPY_DTYPE_U32; break;
					case 8: info.dtype = NPY_DTYPE_U64; break;
					default: npy_ERRMSG(errmsg, "Unsupported unsigned integer data type");
				}
			} else {
				npy_ERRMSG(errmsg, "Unsupported data type");
			}

			if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
				npy_ERRMSG(errmsg, "Unrecognized or unterminated value string for key 'descr'");

		}
		else if (npy_optional(&t,e,"fortran_order")) {
			if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
				npy_ERRMSG(errmsg, "Unrecognized or unterminated dict key");

			if (!npy_consume(&t,e,":"))
				npy_ERRMSG(errmsg, "Expected ':' following key 'fortran_order'");

			if (npy_optional(&t,e,"True"))
				npy_ERRMSG(errmsg, "Fortran order arrays are not supported");

			if (!npy_consume(&t,e,"False"))
				npy_ERRMSG(errmsg, "Expected 'False'");
		}
		else if (npy_optional(&t,e,"shape")) {
			if (!(npy_optional(&t,e,"'") || npy_optional(&t,e,"\"")))
				npy_ERRMSG(errmsg, "Unrecognized or unterminated dict key");

			if (!npy_consume(&t,e,":"))
				npy_ERRMSG(errmsg, "Expected ':' following key 'shape'");

			if (!npy_consume(&t,e,"("))
				npy_ERRMSG(errmsg, "Expected tuple for key 'shape'");

			int64_t val = -1;
			while (val = npy_parse_positive_int(&t,e), val != -1) {
				if (info.dim == 6)
					npy_ERRMSG(errmsg, "Unsupported array (more than 6 dimensions)");
				info.shape[info.dim] = val;
				(info.dim)++;
				if (!npy_optional(&t,e,",")) break;
			}
			if(!npy_consume(&t,e,")")) npy_ERRMSG(errmsg, "Expected ')' in shape");
		}
		else {
			npy_ERRMSG(errmsg, "Unexpected dict key encountered");
		}

		if(i < 2 && !npy_consume(&t,e,",")) npy_ERRMSG(errmsg, "Expected ','");
	}

	(void) npy_optional(&t,e,",");

	if(!npy_consume(&t,e,"}")) npy_ERRMSG(errmsg, "Expected '}'");

	return info;
}

#define npy_ERRMSG2(errmsg_ptr, errmsg_string) do {npy_errmsg((errmsg_ptr), (errmsg_string)); return;} while(0)

NPY_API void
npy_copy_data(NPYInfo info, void *dst, int64_t dst_bytes, const char **errmsg)
{
	int esize = npy_dtype_size(info.dtype);
	if (esize <= 0) npy_ERRMSG2(errmsg, "npy_copy_data: invalid dtype in info");
	int64_t need = npy_nel(info) * (int64_t)esize;
	if (dst_bytes < need) npy_ERRMSG2(errmsg, "npy_copy_data: dst buffer too small");
	if (!dst)             npy_ERRMSG2(errmsg, "npy_copy_data: dst is NULL");
	if (!info.data_ptr)   npy_ERRMSG2(errmsg, "npy_copy_data: info.data_ptr is NULL");

	memcpy(dst, info.data_ptr, (size_t)need);

	if (!info.is_native_byte_order && esize > 1) {
		/* Element-wise byteswap in `dst`. Source is left untouched
		   so this is safe against read-only mmap. */
		uint8_t *p = (uint8_t*)dst;
		for (int64_t i = 0; i < need; i += esize) {
			for (int j = 0; j < esize / 2; j++) {
				uint8_t tmp = p[i + j];
				p[i + j] = p[i + esize - 1 - j];
				p[i + esize - 1 - j] = tmp;
			}
		}
	}
}

// TODO we need a version of this that just encodes to memory
NPY_API void
npy_write_file(FILE *f, void* data, int64_t nbytes, int dim, int64_t *shape, int dtype, const char **errmsg)
// Does not close `f` after completing.
// Returns a static error string, or 0.
{
	char order_mark = '<';
	if(!npy_host_is_little_endian()) 
		order_mark = '>';

	const char *type_string = 0;
	switch(dtype) {
		case NPY_DTYPE_I8:  type_string = "i1"; order_mark='|'; break;
		case NPY_DTYPE_I16: type_string = "i2"; break;
		case NPY_DTYPE_I32: type_string = "i4"; break;
		case NPY_DTYPE_I64: type_string = "i8"; break;

		case NPY_DTYPE_U8:  type_string = "u1"; order_mark='|'; break;
		case NPY_DTYPE_U16: type_string = "u2"; break;
		case NPY_DTYPE_U32: type_string = "u4"; break;
		case NPY_DTYPE_U64: type_string = "u8"; break;

		case NPY_DTYPE_F16: type_string = "f2"; break;
		case NPY_DTYPE_F32: type_string = "f4"; break;
		case NPY_DTYPE_F64: type_string = "f8"; break;

		case NPY_DTYPE_C64:  type_string = "c8";  break;
		case NPY_DTYPE_C128: type_string = "c16"; break;

		default: npy_ERRMSG2(errmsg, "Invalid data type");
	}

	if(dim == 0 || dim > 6) npy_ERRMSG2(errmsg, "Unsupported dimension");

	char fmt_shape[128] = {0};
	char *pos = fmt_shape, *end = fmt_shape+127;
	for (int d = 0; pos < end && d < dim; d++) {
		pos += snprintf(pos, end-pos, "%" PRIi64 , shape[d]);
		if (d != dim-1 || dim == 1) 
			pos += snprintf(pos, end-pos, ", ");
	}

	if (pos > end)
		npy_ERRMSG2(errmsg, "Shape too large");


	char string_header[1024] = {0};
	memset(string_header, '\n', sizeof(string_header));

	int string_header_len =
			snprintf(string_header,
					 1023,
					 "{'descr': '%c%s', 'fortran_order': False, 'shape': (%s)}",
					 order_mark,
					 type_string,
					 fmt_shape);

	uint32_t hdr_len = npy_round_up(10+string_header_len,64)-10;

	if(hdr_len > 1024)
		npy_ERRMSG2(errmsg, "Failed to format header");

	//annoyingly, we need to NOT have a null byte in the string header, or numpy won't load it
	string_header[string_header_len] = '\n';

	uint8_t fixed_header[10] = {0};
	memcpy(fixed_header, "\x93NUMPY", 6);
	fixed_header[6] = 1;
	fixed_header[8] = hdr_len & 0xff;
	fixed_header[9] = (hdr_len >> 8) & 0xff;

	if(1 != fwrite(fixed_header, sizeof(fixed_header), 1, f))
		npy_ERRMSG2(errmsg, "Failed to write the fixed header to the output .npy file");

	if(1 != fwrite(string_header, hdr_len, 1, f))
		npy_ERRMSG2(errmsg, "Failed to write the string header to the output .npy file");

	if(nbytes != (int64_t)fwrite(data, 1, nbytes, f))
		npy_ERRMSG2(errmsg, "Failed to write the array data to the output .npy file");

}

#undef NPYFILES_IMPLEMENTATION
#endif
