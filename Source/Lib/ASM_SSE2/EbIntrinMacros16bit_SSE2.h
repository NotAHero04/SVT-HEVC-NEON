/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifdef __cplusplus
extern "C" {
#endif
#define MACRO_VERTICAL_LUMA_8(ARG1, ARG2, ARG3)\
    simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);

#define MACRO_UNPACK(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11, ARG12, ARG13)\
    ARG10 = simde_mm_unpackhi_epi##ARG1(ARG2, ARG3);\
    ARG2  = simde_mm_unpacklo_epi##ARG1(ARG2, ARG3);\
    ARG11 = simde_mm_unpackhi_epi##ARG1(ARG4, ARG5);\
    ARG4  = simde_mm_unpacklo_epi##ARG1(ARG4, ARG5);\
    ARG12 = simde_mm_unpackhi_epi##ARG1(ARG6, ARG7);\
    ARG6  = simde_mm_unpacklo_epi##ARG1(ARG6, ARG7);\
    ARG13 = simde_mm_unpackhi_epi##ARG1(ARG8, ARG9);\
    ARG8  = simde_mm_unpacklo_epi##ARG1(ARG8, ARG9);

#define MACRO_UNPACK_V2(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11)\
    ARG10 = simde_mm_unpackhi_epi##ARG1(ARG2, ARG3);\
    ARG2  = simde_mm_unpacklo_epi##ARG1(ARG2, ARG3);\
    ARG11 = simde_mm_unpackhi_epi##ARG1(ARG4, ARG5);\
    ARG4  = simde_mm_unpacklo_epi##ARG1(ARG4, ARG5);\
    ARG6  = simde_mm_unpacklo_epi##ARG1(ARG6, ARG7);\
    ARG8  = simde_mm_unpacklo_epi##ARG1(ARG8, ARG9);

#ifdef __cplusplus
}
#endif
