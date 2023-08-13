/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifdef __cplusplus
extern "C" {
#endif

#define MACRO_VERTICAL_LUMA_8(A, B, C)\
    simde_mm_storel_epi64((simde__m128i*)predictionPtr, simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storel_epi64((simde__m128i*)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storel_epi64((simde__m128i*)(predictionPtr + 2*pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storel_epi64((simde__m128i*)(predictionPtr + 3*pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1);

#define MACRO_VERTICAL_LUMA_16(A, B, C)\
    simde_mm_storeu_si128((simde__m128i*)predictionPtr, simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storeu_si128((simde__m128i*)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storeu_si128((simde__m128i*)(predictionPtr + 2*pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    simde_mm_storeu_si128((simde__m128i*)(predictionPtr + 3*pStride), simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1);

#ifdef __cplusplus
}
#endif
