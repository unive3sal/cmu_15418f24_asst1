#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// implementation of absolute value using 15418 instrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// Set mask according to predicate
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	float result = 1.f;
	int y = exponents[i];
	float xpower = x;
	while (y > 0) {
	    if (y & 0x1) {
			result *= xpower;
		}
	    xpower = xpower * xpower;
	    y >>= 1;
	}
	if (result > 4.18f) {
	    result = 4.18f;
	}
	output[i] = result;
    }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __cmu418_mask mask;
        __cmu418_vec_float x, result, xpower;
        __cmu418_vec_int y, ones, zeros;

        mask = N - i < VECTOR_WIDTH ? _cmu418_init_ones(N - i) : _cmu418_init_ones();
        _cmu418_vload_float(x, values + i, mask);
        _cmu418_vset_float(result, 1.f, mask);
        _cmu418_vload_int(y, exponents + i, mask);
        _cmu418_vload_float(xpower, values + i, mask);
        _cmu418_vset_int(ones, 1, mask);
        _cmu418_vset_int(zeros, 0, mask);

        __cmu418_mask exp_cond_mask;
        _cmu418_vgt_int(exp_cond_mask, y, zeros, mask);
        while (_cmu418_cntbits(exp_cond_mask) > 0) {
            // if (y & 0x1)
            __cmu418_vec_int bit_and_res;
            _cmu418_vbitand_int(bit_and_res, y, ones, mask);
            __cmu418_mask bit_and_mask;
            _cmu418_vgt_int(bit_and_mask, bit_and_res, zeros, mask);
            _cmu418_vmult_float(result, result, xpower, bit_and_mask);

            _cmu418_vmult_float(xpower, xpower, xpower, exp_cond_mask);

            _cmu418_vshiftright_int(y, y, ones, exp_cond_mask);
            _cmu418_vgt_int(exp_cond_mask, y, zeros, mask);
        }
        __cmu418_vec_float th = _cmu418_vset_float(4.18f);
        __cmu418_mask result_mask;
        _cmu418_vgt_float(result_mask, result, th, mask);
        _cmu418_vset_float(result, 4.18f, result_mask);

        _cmu418_vstore_float(output + i, result, mask);
    }
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
	sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
    __cmu418_vec_float sum = _cmu418_vset_float(0.f);
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __cmu418_vec_float vals;
        __cmu418_mask mask = _cmu418_init_ones();
        _cmu418_vload_float(vals, values + i, mask);
        _cmu418_vadd_float(sum, sum, vals, mask);
    }

    float ans = 0.f;
    __cmu418_mask mask = _cmu418_init_ones(1);
    unsigned int cnt = VECTOR_WIDTH >> 1;
    while (cnt > 0) {
        _cmu418_hadd_float(sum, sum);
        _cmu418_interleave_float(sum, sum);
        cnt >>= 1;
    }
    _cmu418_vstore_float(&ans, sum, mask);
	return ans;
}
