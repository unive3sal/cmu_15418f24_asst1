#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <assert.h>
#include <stdint.h>

extern void saxpySerial(int N,
			float scale,
			float X[],
			float Y[],
			float result[]);


void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Replace this code with ones that make use of the streaming instructions
    const int VECTOR_WIDTH = sizeof(__m128) / sizeof(float);
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __m128 x_4 = _mm_load_ps(X);
        __m128 y_4 = _mm_load_ps(X);
        __m128 scale_4 = _mm_set1_ps(scale);

        __m128 scale_res_4 = _mm_mul_ps(scale_4, x_4);
        __m128 result_4 = _mm_add_ps(scale_res_4, y_4);
        _mm_store_ps(result + i, result_4);
    }
}

