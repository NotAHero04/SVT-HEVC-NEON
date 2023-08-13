/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbPictureOperators_SSE4_1.h"
#include "../../../simde/simde/x86/sse4.1.h"


EB_U64 Compute8x8Satd_SSE4(
    EB_S16 *diff)       // input parameter, diff samples Ptr
{
    EB_U64 satdBlock8x8 = 0;
    EB_S16 m2[8][8];

    EB_U32 j, jj;
    simde__m128i s0, s1, s2, s3, s4, s5, s6, s7, s9, s10, s11, s12;
    simde__m128i s8 = simde_mm_setzero_si128();
    simde__m128i sum01Neg, sum01Pos, sum23Neg, sum23Pos, sum45Neg, sum45Pos, sum67Neg, sum67Pos;
    simde__m128i sum0to3Pos, sum4to7Pos, sum0to3Neg, sum4to7Neg, diff0to3Pos, diff4to7Pos, diff0to3Neg, diff4to7Neg;
    simde__m128i sum0, sum1, difference0, difference1;

    for (j = 0; j < 8; j += 2)
    {
        jj = j << 3;
        s0 = simde_mm_loadu_si128((simde__m128i*)(diff + jj));
        s10 = simde_mm_loadu_si128((simde__m128i*)(diff + 8 + jj));

        sum0 = simde_mm_hadd_epi16(s0, s8);
        sum1 = simde_mm_hadd_epi16(s10, s8);

        difference0 = simde_mm_hsub_epi16(s0, s8);
        difference1 = simde_mm_hsub_epi16(s10, s8);

        // m2[j][0]
        // diff[jj] + diff[jj + 4] + diff[jj + 2] + diff[jj + 6] + diff[jj + 1] + diff[jj + 5] + diff[jj + 3] + diff[jj + 7]
        // diff[jj] + diff[jj + 1] + diff[jj + 2] + diff[jj + 3] + diff[jj + 4] + diff[jj + 5] + diff[jj + 6] + diff[jj + 7]
        s1 = simde_mm_hadd_epi16(sum0, sum1);
        s1 = simde_mm_hadd_epi16(s1, s8);
        m2[j][0] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][0] = simde_mm_extract_epi16(s1, 2);


        //m2[j][1]
        //diff[jj] + diff[jj + 4] + diff[jj + 2] + diff[jj + 6] - diff[jj + 1] - diff[jj + 5] - diff[jj + 3] - diff[jj + 7]
        //diff[jj] - diff[jj + 1] + diff[jj + 2] - diff[jj + 3] + diff[jj + 4] - diff[jj + 5] + diff[jj + 6] - diff[jj + 7]
        //(diff[jj] - diff[jj + 1]) + (diff[jj + 2] - diff[jj + 3]) + (diff[jj + 4] - diff[jj + 5]) + (diff[jj + 6] - diff[jj + 7])
        s1 = simde_mm_hadd_epi16(difference0, difference1);
        s1 = simde_mm_hadd_epi16(s1, s8);
        m2[j][1] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][1] = simde_mm_extract_epi16(s1, 2);

        //m2[j][2]
        //diff[jj] + diff[jj + 4] - diff[jj + 2] - diff[jj + 6] + diff[jj + 1] + diff[jj + 5] - diff[jj + 3] - diff[jj + 7]
        //diff[jj] + diff[jj + 1] - diff[jj + 2] - diff[jj + 3] + diff[jj + 4] + diff[jj + 5] - diff[jj + 6] - diff[jj + 7]
        s1 = simde_mm_hsub_epi16(sum0, sum1);
        s1 = simde_mm_hadd_epi16(s1, s8);
        m2[j][2] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][2] = simde_mm_extract_epi16(s1, 2);

        //m2[j][3]
        //diff[jj] + diff[jj + 4] - diff[jj + 2] - diff[jj + 6] - diff[jj + 1] - diff[jj + 5] + diff[jj + 3] + diff[jj + 7]
        //diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] + diff[jj + 4] - diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
        //diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] + diff[jj + 4] - diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
        s1 = simde_mm_hsub_epi16(difference0, difference1);
        s1 = simde_mm_hadd_epi16(s1, s8);
        m2[j][3] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][3] = simde_mm_extract_epi16(s1, 2);

        //m2[j][4]
        //diff[jj] - diff[jj + 4] + diff[jj + 2] - diff[jj + 6] + diff[jj + 1] - diff[jj + 5] + diff[jj + 3] - diff[jj + 7]
        //diff[jj] + diff[jj + 1] + diff[jj + 2] + diff[jj + 3] - diff[jj + 4] - diff[jj + 5] - diff[jj + 6] - diff[jj + 7]
        s1 = simde_mm_hadd_epi16(sum0, sum1);
        s1 = simde_mm_hsub_epi16(s1, s8);
        m2[j][4] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][4] = simde_mm_extract_epi16(s1, 2);

        //m2[j][5]
        //m1[j][4] - m1[j][5]
        //diff[jj] - diff[jj + 1] + diff[jj + 2] - diff[jj + 3] - diff[jj + 4]  + diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
        s1 = simde_mm_hadd_epi16(difference0, difference1);
        s1 = simde_mm_hsub_epi16(s1, s8);
        m2[j][5] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][5] = simde_mm_extract_epi16(s1, 2);

        //m2[j][6]
        //diff[jj] - diff[jj + 4] - diff[jj + 2] + diff[jj + 6] + diff[jj + 1] - diff[jj + 5] - diff[jj + 3] + diff[jj + 7]
        //diff[jj] + diff[jj + 1] - diff[jj + 2] - diff[jj + 3] - diff[jj + 4] - diff[jj + 5] + diff[jj + 6] + diff[jj + 7]

        s1 = simde_mm_hsub_epi16(sum0, sum1);
        s1 = simde_mm_hsub_epi16(s1, s8);
        m2[j][6] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][6] = simde_mm_extract_epi16(s1, 2);

        //m2[j][7]
        //diff[jj] - diff[jj + 4] - diff[jj + 2] + diff[jj + 6] - diff[jj + 1] + diff[jj + 5] + diff[jj + 3] - diff[jj + 7]
        //diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] - diff[jj + 4] + diff[jj + 5] + diff[jj + 6] - diff[jj + 7]
        s1 = simde_mm_hsub_epi16(difference0, difference1);
        s1 = simde_mm_hsub_epi16(s1, s8);
        m2[j][7] = simde_mm_extract_epi16(s1, 0);
        m2[j + 1][7] = simde_mm_extract_epi16(s1, 2);
    }

    // Vertical
    s0 = simde_mm_loadu_si128((simde__m128i*)(m2[0]));
    s1 = simde_mm_loadu_si128((simde__m128i*)(m2[1]));
    s2 = simde_mm_loadu_si128((simde__m128i*)(m2[2]));
    s3 = simde_mm_loadu_si128((simde__m128i*)(m2[3]));
    s4 = simde_mm_loadu_si128((simde__m128i*)(m2[4]));
    s5 = simde_mm_loadu_si128((simde__m128i*)(m2[5]));
    s6 = simde_mm_loadu_si128((simde__m128i*)(m2[6]));
    s7 = simde_mm_loadu_si128((simde__m128i*)(m2[7]));

    sum01Pos = simde_mm_add_epi16(s0, s1);
    sum23Pos = simde_mm_add_epi16(s2, s3);
    sum45Pos = simde_mm_add_epi16(s4, s5);
    sum67Pos = simde_mm_add_epi16(s6, s7);

    sum01Neg = simde_mm_sub_epi16(s0, s1);
    sum23Neg = simde_mm_sub_epi16(s2, s3);
    sum45Neg = simde_mm_sub_epi16(s4, s5);
    sum67Neg = simde_mm_sub_epi16(s6, s7);

    sum0to3Pos = simde_mm_add_epi16(sum01Pos, sum23Pos);
    sum4to7Pos = simde_mm_add_epi16(sum45Pos, sum67Pos);
    diff0to3Pos = simde_mm_sub_epi16(sum01Pos, sum23Pos);
    diff4to7Pos = simde_mm_sub_epi16(sum45Pos, sum67Pos);

    sum0to3Neg = simde_mm_add_epi16(sum01Neg, sum23Neg);
    sum4to7Neg = simde_mm_add_epi16(sum45Neg, sum67Neg);
    diff0to3Neg = simde_mm_sub_epi16(sum01Neg, sum23Neg);
    diff4to7Neg = simde_mm_sub_epi16(sum45Neg, sum67Neg);

    //m2[0][i] = m1[0][i] + m1[1][i]
    //m2[0][i] = m3[0][i] + m3[2][i] + m3[1][i] + m3[3][i]
    //m2[0][i] = m2[0][i] + m2[4][i] + m2[2][i] + m2[6][i] + m2[1][i] + m2[5][i] + m2[3][i] + m2[7][i]
    //m2[0][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] + m2[4][i] + m2[5][i] + m2[6][i] + m2[7][i]
    s9 = simde_mm_add_epi16(sum0to3Pos, sum4to7Pos);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);


    //m2[1][i] = m1[0][i] - m1[1][i]
    //m2[1][i] = m3[0][i] + m3[2][i] -(m3[1][i] + m3[3][i])
    //m2[1][i] = m2[0][i] + m2[4][i] + m2[2][i] + m2[6][i] -(m2[1][i] + m2[5][i] + m2[3][i] + m2[7][i])
    //m2[1][i] = m2[0][i] - m2[1][i] + m2[2][i] - m2[3][i] + m2[4][i] - m2[5][i] + m2[6][i] - m2[7][i]
    s9 = simde_mm_add_epi16(sum0to3Neg, sum4to7Neg);
    s9 = simde_mm_abs_epi16(s9);
    s12 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s12 = simde_mm_add_epi32(s12, s11);
    s12 = simde_mm_hadd_epi32(s12, s8);
    s12 = simde_mm_hadd_epi32(s12, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[2][i] = m1[2][i] + m1[3][i]
    //m2[2][i] = m3[0][i] - m3[2][i] + m3[1][i] - m3[3][i]
    //m2[2][i] = m2[0][i] + m2[4][i] - (m2[2][i] + m2[6][i]) + m2[1][i] + m2[5][i] - (m2[3][i] + m2[7][i])
    //m2[2][i] = m2[0][i] + m2[1][i] - m2[2][i] - m2[3][i] + m2[4][i] + m2[5][i] - m2[6][i] - m2[7][i]
    //m2[2][i] = m2[0][i] + m2[1][i] - (m2[2][i] + m2[3][i]) + m2[4][i] + m2[5][i] - (m2[6][i] + m2[7][i])
    s9 = simde_mm_add_epi16(diff0to3Pos, diff4to7Pos);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[3][i] = m1[2][i] - m1[3][i]
    //m2[3][i] = m3[0][i] - m3[2][i] - (m3[1][i] - m3[3][i])
    //m2[3][i] = m2[0][i] + m2[4][i] - (m2[2][i] + m2[6][i]) - (m2[1][i] + m2[5][i] - m2[3][i] - m2[7][i])
    //m2[3][i] = m2[0][i] - m2[1][i] - m2[2][i] + m2[3][i] + m2[4][i] - m2[5][i] - m2[6][i] + m2[7][i]
    //m2[3][i] = m2[0][i] - m2[1][i] - (m2[2][i] - m2[3][i]) + (m2[4][i] - m2[5][i]) - (m2[6][i] - m2[7][i])
    s9 = simde_mm_add_epi16(diff0to3Neg, diff4to7Neg);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[4][i] = m1[4][i] + m1[5][i]
    //m2[4][i] = m3[4][i] + m3[6][i] + m3[5][i] + m3[7][i]
    //m2[4][i] = m2[0][i] - m2[4][i] + m2[2][i] - m2[6][i] + m2[1][i] - m2[5][i] + m2[3][i] - m2[7][i]
    //m2[4][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] - m2[4][i] - m2[5][i] - m2[6][i] - m2[7][i]
    //m2[4][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] - ( (m2[4][i] + m2[5][i]) + (m2[6][i] + m2[7][i]) )
    s9 = simde_mm_sub_epi16(sum0to3Pos, sum4to7Pos);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[5][i] = m1[4][i] - m1[5][i]
    //m2[5][i] = m3[4][i] + m3[6][i] - (m3[5][i] + m3[7][i])
    //m2[5][i] = m2[0][i] - m2[4][i] + m2[2][i] - m2[6][i] - (m2[1][i] - m2[5][i] + m2[3][i] - m2[7][i])
    //m2[5][i] = m2[0][i] - m2[1][i] + m2[2][i] - m2[3][i] - m2[4][i] + m2[5][i] - m2[6][i] + m2[7][i]
    //m2[5][i] = m2[0][i] - m2[1][i] + (m2[2][i] - m2[3][i]) - ( (m2[4][i] - m2[5][i]) + (m2[6][i] - m2[7][i]) )
    s9 = simde_mm_sub_epi16(sum0to3Neg, sum4to7Neg);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[6][i] = m1[6][i] + m1[7][i]
    //m2[6][i] = m3[4][i] - m3[6][i] + m3[5][i] - m3[7][i]
    //m2[6][i] = m2[0][i] - m2[4][i] - (m2[2][i] - m2[6][i]) + m2[1][i] - m2[5][i] - (m2[3][i] - m2[7][i])
    //m2[6][i] = m2[0][i] + m2[1][i] - m2[2][i] - m2[3][i] - m2[4][i] - m2[5][i] + m2[6][i] + m2[7][i]
    //m2[6][i] = (m2[0][i] + m2[1][i]) - (m2[2][i] + m2[3][i]) - ( (m2[4][i] + m2[5][i]) - (m2[6][i] + m2[7][i]) )
    s9 = simde_mm_sub_epi16(diff0to3Pos, diff4to7Pos);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    //m2[7][i] = m1[6][i] - m1[7][i]
    //m2[7][i] = m3[4][i] - m3[6][i] - (m3[5][i] - m3[7][i])
    //m2[7][i] = m2[0][i] - m2[4][i] - (m2[2][i] - m2[6][i]) - ((m2[1][i] - m2[5][i]) - (m2[3][i] - m2[7][i]))
    //m2[7][i] = (m2[0][i] - m2[1][i]) - (m2[2][i] - m2[3][i]) - ( (m2[4][i] - m2[5][i]) - (m2[6][i] - m2[7][i]) )
    s9 = simde_mm_sub_epi16(diff0to3Neg, diff4to7Neg);
    s9 = simde_mm_abs_epi16(s9);
    s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
    s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
    s10 = simde_mm_add_epi32(s10, s11);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s10 = simde_mm_hadd_epi32(s10, s8);
    s12 = simde_mm_add_epi32(s10, s12);

    satdBlock8x8 = (EB_U64)simde_mm_extract_epi32(s12, 0);

    satdBlock8x8 = ((satdBlock8x8 + 2) >> 2);

    return satdBlock8x8;
}

EB_U64 Compute8x8Satd_U8_SSE4(
	EB_U8  *src,       // input parameter, diff samples Ptr
	EB_U64 *dcValue,
	EB_U32  srcStride)
{
	EB_U64 satdBlock8x8 = 0;
	EB_S16 m2[8][8];

	EB_U32 j;
	simde__m128i s0, s1, s2, s3, s4, s5, s6, s7, s9, s10, s11, s12;
	simde__m128i s8 = simde_mm_setzero_si128();
	simde__m128i sum01Neg, sum01Pos, sum23Neg, sum23Pos, sum45Neg, sum45Pos, sum67Neg, sum67Pos;
	simde__m128i sum0to3Pos, sum4to7Pos, sum0to3Neg, sum4to7Neg, diff0to3Pos, diff4to7Pos, diff0to3Neg, diff4to7Neg;
	simde__m128i sum0, sum1, difference0, difference1;

	for (j = 0; j < 8; j += 2)
	{
		s0 = simde_mm_loadl_epi64((simde__m128i*)(src + (j *srcStride)));
		s10 = simde_mm_loadl_epi64((simde__m128i*)(src + ((j + 1) *srcStride)));
		s10 = simde_mm_unpacklo_epi8(s10, simde_mm_setzero_si128());
		s0 = simde_mm_unpacklo_epi8(s0, simde_mm_setzero_si128());

		sum0 = simde_mm_hadd_epi16(s0, s8);
		sum1 = simde_mm_hadd_epi16(s10, s8);

		difference0 = simde_mm_hsub_epi16(s0, s8);
		difference1 = simde_mm_hsub_epi16(s10, s8);

		// m2[j][0]
		// diff[jj] + diff[jj + 4] + diff[jj + 2] + diff[jj + 6] + diff[jj + 1] + diff[jj + 5] + diff[jj + 3] + diff[jj + 7]
		// diff[jj] + diff[jj + 1] + diff[jj + 2] + diff[jj + 3] + diff[jj + 4] + diff[jj + 5] + diff[jj + 6] + diff[jj + 7]
		s1 = simde_mm_hadd_epi16(sum0, sum1);
		s1 = simde_mm_hadd_epi16(s1, s8);
		m2[j][0] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][0] = simde_mm_extract_epi16(s1, 2);


		//m2[j][1]
		//diff[jj] + diff[jj + 4] + diff[jj + 2] + diff[jj + 6] - diff[jj + 1] - diff[jj + 5] - diff[jj + 3] - diff[jj + 7]
		//diff[jj] - diff[jj + 1] + diff[jj + 2] - diff[jj + 3] + diff[jj + 4] - diff[jj + 5] + diff[jj + 6] - diff[jj + 7]
		//(diff[jj] - diff[jj + 1]) + (diff[jj + 2] - diff[jj + 3]) + (diff[jj + 4] - diff[jj + 5]) + (diff[jj + 6] - diff[jj + 7])
		s1 = simde_mm_hadd_epi16(difference0, difference1);
		s1 = simde_mm_hadd_epi16(s1, s8);
		m2[j][1] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][1] = simde_mm_extract_epi16(s1, 2);

		//m2[j][2]
		//diff[jj] + diff[jj + 4] - diff[jj + 2] - diff[jj + 6] + diff[jj + 1] + diff[jj + 5] - diff[jj + 3] - diff[jj + 7]
		//diff[jj] + diff[jj + 1] - diff[jj + 2] - diff[jj + 3] + diff[jj + 4] + diff[jj + 5] - diff[jj + 6] - diff[jj + 7]
		s1 = simde_mm_hsub_epi16(sum0, sum1);
		s1 = simde_mm_hadd_epi16(s1, s8);
		m2[j][2] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][2] = simde_mm_extract_epi16(s1, 2);

		//m2[j][3]
		//diff[jj] + diff[jj + 4] - diff[jj + 2] - diff[jj + 6] - diff[jj + 1] - diff[jj + 5] + diff[jj + 3] + diff[jj + 7]
		//diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] + diff[jj + 4] - diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
		//diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] + diff[jj + 4] - diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
		s1 = simde_mm_hsub_epi16(difference0, difference1);
		s1 = simde_mm_hadd_epi16(s1, s8);
		m2[j][3] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][3] = simde_mm_extract_epi16(s1, 2);

		//m2[j][4]
		//diff[jj] - diff[jj + 4] + diff[jj + 2] - diff[jj + 6] + diff[jj + 1] - diff[jj + 5] + diff[jj + 3] - diff[jj + 7]
		//diff[jj] + diff[jj + 1] + diff[jj + 2] + diff[jj + 3] - diff[jj + 4] - diff[jj + 5] - diff[jj + 6] - diff[jj + 7]
		s1 = simde_mm_hadd_epi16(sum0, sum1);
		s1 = simde_mm_hsub_epi16(s1, s8);
		m2[j][4] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][4] = simde_mm_extract_epi16(s1, 2);

		//m2[j][5]
		//m1[j][4] - m1[j][5]
		//diff[jj] - diff[jj + 1] + diff[jj + 2] - diff[jj + 3] - diff[jj + 4]  + diff[jj + 5] - diff[jj + 6] + diff[jj + 7]
		s1 = simde_mm_hadd_epi16(difference0, difference1);
		s1 = simde_mm_hsub_epi16(s1, s8);
		m2[j][5] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][5] = simde_mm_extract_epi16(s1, 2);

		//m2[j][6]
		//diff[jj] - diff[jj + 4] - diff[jj + 2] + diff[jj + 6] + diff[jj + 1] - diff[jj + 5] - diff[jj + 3] + diff[jj + 7]
		//diff[jj] + diff[jj + 1] - diff[jj + 2] - diff[jj + 3] - diff[jj + 4] - diff[jj + 5] + diff[jj + 6] + diff[jj + 7]

		s1 = simde_mm_hsub_epi16(sum0, sum1);
		s1 = simde_mm_hsub_epi16(s1, s8);
		m2[j][6] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][6] = simde_mm_extract_epi16(s1, 2);

		//m2[j][7]
		//diff[jj] - diff[jj + 4] - diff[jj + 2] + diff[jj + 6] - diff[jj + 1] + diff[jj + 5] + diff[jj + 3] - diff[jj + 7]
		//diff[jj] - diff[jj + 1] - diff[jj + 2] + diff[jj + 3] - diff[jj + 4] + diff[jj + 5] + diff[jj + 6] - diff[jj + 7]
		s1 = simde_mm_hsub_epi16(difference0, difference1);
		s1 = simde_mm_hsub_epi16(s1, s8);
		m2[j][7] = simde_mm_extract_epi16(s1, 0);
		m2[j + 1][7] = simde_mm_extract_epi16(s1, 2);
	}

	// Vertical
	s0 = simde_mm_loadu_si128((simde__m128i*)(m2[0]));
	s1 = simde_mm_loadu_si128((simde__m128i*)(m2[1]));
	s2 = simde_mm_loadu_si128((simde__m128i*)(m2[2]));
	s3 = simde_mm_loadu_si128((simde__m128i*)(m2[3]));
	s4 = simde_mm_loadu_si128((simde__m128i*)(m2[4]));
	s5 = simde_mm_loadu_si128((simde__m128i*)(m2[5]));
	s6 = simde_mm_loadu_si128((simde__m128i*)(m2[6]));
	s7 = simde_mm_loadu_si128((simde__m128i*)(m2[7]));

	sum01Pos = simde_mm_add_epi16(s0, s1);
	sum23Pos = simde_mm_add_epi16(s2, s3);
	sum45Pos = simde_mm_add_epi16(s4, s5);
	sum67Pos = simde_mm_add_epi16(s6, s7);

	sum01Neg = simde_mm_sub_epi16(s0, s1);
	sum23Neg = simde_mm_sub_epi16(s2, s3);
	sum45Neg = simde_mm_sub_epi16(s4, s5);
	sum67Neg = simde_mm_sub_epi16(s6, s7);

	sum0to3Pos = simde_mm_add_epi16(sum01Pos, sum23Pos);
	sum4to7Pos = simde_mm_add_epi16(sum45Pos, sum67Pos);
	diff0to3Pos = simde_mm_sub_epi16(sum01Pos, sum23Pos);
	diff4to7Pos = simde_mm_sub_epi16(sum45Pos, sum67Pos);

	sum0to3Neg = simde_mm_add_epi16(sum01Neg, sum23Neg);
	sum4to7Neg = simde_mm_add_epi16(sum45Neg, sum67Neg);
	diff0to3Neg = simde_mm_sub_epi16(sum01Neg, sum23Neg);
	diff4to7Neg = simde_mm_sub_epi16(sum45Neg, sum67Neg);

	//m2[0][i] = m1[0][i] + m1[1][i]
	//m2[0][i] = m3[0][i] + m3[2][i] + m3[1][i] + m3[3][i]
	//m2[0][i] = m2[0][i] + m2[4][i] + m2[2][i] + m2[6][i] + m2[1][i] + m2[5][i] + m2[3][i] + m2[7][i]
	//m2[0][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] + m2[4][i] + m2[5][i] + m2[6][i] + m2[7][i]
	s9 = simde_mm_add_epi16(sum0to3Pos, sum4to7Pos);
	s9 = simde_mm_abs_epi16(s9);
	*dcValue += simde_mm_extract_epi16(s9, 0);

	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);


	//m2[1][i] = m1[0][i] - m1[1][i]
	//m2[1][i] = m3[0][i] + m3[2][i] -(m3[1][i] + m3[3][i])
	//m2[1][i] = m2[0][i] + m2[4][i] + m2[2][i] + m2[6][i] -(m2[1][i] + m2[5][i] + m2[3][i] + m2[7][i])
	//m2[1][i] = m2[0][i] - m2[1][i] + m2[2][i] - m2[3][i] + m2[4][i] - m2[5][i] + m2[6][i] - m2[7][i]
	s9 = simde_mm_add_epi16(sum0to3Neg, sum4to7Neg);
	s9 = simde_mm_abs_epi16(s9);
	s12 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s12 = simde_mm_add_epi32(s12, s11);
	s12 = simde_mm_hadd_epi32(s12, s8);
	s12 = simde_mm_hadd_epi32(s12, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[2][i] = m1[2][i] + m1[3][i]
	//m2[2][i] = m3[0][i] - m3[2][i] + m3[1][i] - m3[3][i]
	//m2[2][i] = m2[0][i] + m2[4][i] - (m2[2][i] + m2[6][i]) + m2[1][i] + m2[5][i] - (m2[3][i] + m2[7][i])
	//m2[2][i] = m2[0][i] + m2[1][i] - m2[2][i] - m2[3][i] + m2[4][i] + m2[5][i] - m2[6][i] - m2[7][i]
	//m2[2][i] = m2[0][i] + m2[1][i] - (m2[2][i] + m2[3][i]) + m2[4][i] + m2[5][i] - (m2[6][i] + m2[7][i])
	s9 = simde_mm_add_epi16(diff0to3Pos, diff4to7Pos);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[3][i] = m1[2][i] - m1[3][i]
	//m2[3][i] = m3[0][i] - m3[2][i] - (m3[1][i] - m3[3][i])
	//m2[3][i] = m2[0][i] + m2[4][i] - (m2[2][i] + m2[6][i]) - (m2[1][i] + m2[5][i] - m2[3][i] - m2[7][i])
	//m2[3][i] = m2[0][i] - m2[1][i] - m2[2][i] + m2[3][i] + m2[4][i] - m2[5][i] - m2[6][i] + m2[7][i]
	//m2[3][i] = m2[0][i] - m2[1][i] - (m2[2][i] - m2[3][i]) + (m2[4][i] - m2[5][i]) - (m2[6][i] - m2[7][i])
	s9 = simde_mm_add_epi16(diff0to3Neg, diff4to7Neg);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[4][i] = m1[4][i] + m1[5][i]
	//m2[4][i] = m3[4][i] + m3[6][i] + m3[5][i] + m3[7][i]
	//m2[4][i] = m2[0][i] - m2[4][i] + m2[2][i] - m2[6][i] + m2[1][i] - m2[5][i] + m2[3][i] - m2[7][i]
	//m2[4][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] - m2[4][i] - m2[5][i] - m2[6][i] - m2[7][i]
	//m2[4][i] = m2[0][i] + m2[1][i] + m2[2][i] + m2[3][i] - ( (m2[4][i] + m2[5][i]) + (m2[6][i] + m2[7][i]) )
	s9 = simde_mm_sub_epi16(sum0to3Pos, sum4to7Pos);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[5][i] = m1[4][i] - m1[5][i]
	//m2[5][i] = m3[4][i] + m3[6][i] - (m3[5][i] + m3[7][i])
	//m2[5][i] = m2[0][i] - m2[4][i] + m2[2][i] - m2[6][i] - (m2[1][i] - m2[5][i] + m2[3][i] - m2[7][i])
	//m2[5][i] = m2[0][i] - m2[1][i] + m2[2][i] - m2[3][i] - m2[4][i] + m2[5][i] - m2[6][i] + m2[7][i]
	//m2[5][i] = m2[0][i] - m2[1][i] + (m2[2][i] - m2[3][i]) - ( (m2[4][i] - m2[5][i]) + (m2[6][i] - m2[7][i]) )
	s9 = simde_mm_sub_epi16(sum0to3Neg, sum4to7Neg);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[6][i] = m1[6][i] + m1[7][i]
	//m2[6][i] = m3[4][i] - m3[6][i] + m3[5][i] - m3[7][i]
	//m2[6][i] = m2[0][i] - m2[4][i] - (m2[2][i] - m2[6][i]) + m2[1][i] - m2[5][i] - (m2[3][i] - m2[7][i])
	//m2[6][i] = m2[0][i] + m2[1][i] - m2[2][i] - m2[3][i] - m2[4][i] - m2[5][i] + m2[6][i] + m2[7][i]
	//m2[6][i] = (m2[0][i] + m2[1][i]) - (m2[2][i] + m2[3][i]) - ( (m2[4][i] + m2[5][i]) - (m2[6][i] + m2[7][i]) )
	s9 = simde_mm_sub_epi16(diff0to3Pos, diff4to7Pos);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	//m2[7][i] = m1[6][i] - m1[7][i]
	//m2[7][i] = m3[4][i] - m3[6][i] - (m3[5][i] - m3[7][i])
	//m2[7][i] = m2[0][i] - m2[4][i] - (m2[2][i] - m2[6][i]) - ((m2[1][i] - m2[5][i]) - (m2[3][i] - m2[7][i]))
	//m2[7][i] = (m2[0][i] - m2[1][i]) - (m2[2][i] - m2[3][i]) - ( (m2[4][i] - m2[5][i]) - (m2[6][i] - m2[7][i]) )
	s9 = simde_mm_sub_epi16(diff0to3Neg, diff4to7Neg);
	s9 = simde_mm_abs_epi16(s9);
	s10 = simde_mm_unpacklo_epi16(s9, simde_mm_setzero_si128());
	s11 = simde_mm_unpackhi_epi16(s9, simde_mm_setzero_si128());
	s10 = simde_mm_add_epi32(s10, s11);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s10 = simde_mm_hadd_epi32(s10, s8);
	s12 = simde_mm_add_epi32(s10, s12);

	satdBlock8x8 = (EB_U64)simde_mm_extract_epi32(s12, 0);

	satdBlock8x8 = ((satdBlock8x8 + 2) >> 2);

	return satdBlock8x8;
}

EB_U64 SpatialFullDistortionKernel4x4_SSSE3_INTRIN(
    EB_U8   *input,
    EB_U32   inputStride,
    EB_U8   *recon,
    EB_U32   reconStride,
    EB_U32   areaWidth,
    EB_U32   areaHeight)
{
    EB_U64  spatialDistortion = 0;
    EB_S32 rowCount;
    simde__m128i sum = simde_mm_setzero_si128();

    rowCount = 4;
    do
    {
        simde__m128i x0;
        simde__m128i y0;
        x0 = simde_mm_setr_epi32(*((EB_U32 *)input), 0, 0, 0);
        y0 = simde_mm_setr_epi32(*((EB_U32 *)recon), 0, 0, 0);
        x0 = simde_mm_cvtepu8_epi16(x0);
        y0 = simde_mm_cvtepu8_epi16(y0);

        x0 = simde_mm_sub_epi16(x0, y0);
        x0 = simde_mm_madd_epi16(x0, x0);
        sum = simde_mm_add_epi32(sum, x0);
        input += inputStride;
        recon += reconStride;
    } while (--rowCount);

    sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0xe1)); // 11100001
    spatialDistortion = simde_mm_extract_epi32(sum, 0);

    (void)areaWidth;
    (void)areaHeight;
    return spatialDistortion;

};

EB_U64 SpatialFullDistortionKernel8x8_SSSE3_INTRIN(
    EB_U8   *input,
    EB_U32   inputStride,
    EB_U8   *recon,
    EB_U32   reconStride,
    EB_U32   areaWidth,
    EB_U32   areaHeight)
{
    EB_U64  spatialDistortion = 0;
    EB_S32 rowCount;
    simde__m128i sum = simde_mm_setzero_si128();

    rowCount = 8;
    do
    {
        simde__m128i x0;
        simde__m128i y0;
        x0 = simde_mm_loadl_epi64/*simde_mm_loadu_si128*/((simde__m128i *)(input + 0x00));
        y0 = simde_mm_loadl_epi64((simde__m128i *)(recon + 0x00));
        x0 = simde_mm_cvtepu8_epi16(x0);
        y0 = simde_mm_cvtepu8_epi16(y0);

        x0 = simde_mm_sub_epi16(x0, y0);
        x0 = simde_mm_madd_epi16(x0, x0);
        sum = simde_mm_add_epi32(sum, x0);

        input += inputStride;
        recon += reconStride;

     } while (--rowCount);

        sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
        sum = simde_mm_unpacklo_epi32(sum, sum);
        sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
        spatialDistortion = simde_mm_extract_epi32(sum, 0);

        (void)areaWidth;
        (void)areaHeight;
        return spatialDistortion;

};

EB_U64 SpatialFullDistortionKernel16MxN_SSSE3_INTRIN(
    EB_U8   *input,
    EB_U32   inputStride,
    EB_U8   *recon,
    EB_U32   reconStride,
    EB_U32   areaWidth,
    EB_U32   areaHeight)
{
    EB_U64  spatialDistortion = 0;
    EB_S32 rowCount, colCount;
    simde__m128i sum = simde_mm_setzero_si128();
    simde__m128i x0, y0, x0_L, x0_H, y0_L, y0_H;


    colCount = areaWidth;
    do
    {
        EB_U8 *coeffTemp = input;
        EB_U8 *reconCoeffTemp = recon;

        rowCount = areaHeight;
        do
        {
            x0 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x00));
            y0 = simde_mm_loadu_si128((simde__m128i *)(reconCoeffTemp + 0x00));
            x0_L = simde_mm_unpacklo_epi8(x0, simde_mm_setzero_si128());
            x0_H = simde_mm_unpackhi_epi8(x0, simde_mm_setzero_si128());
            y0_L = simde_mm_unpacklo_epi8(y0, simde_mm_setzero_si128());
            y0_H = simde_mm_unpackhi_epi8(y0, simde_mm_setzero_si128());

            x0_L = simde_mm_sub_epi16(x0_L, y0_L);
            x0_H = simde_mm_sub_epi16(x0_H, y0_H);

            x0_L = simde_mm_madd_epi16(x0_L, x0_L);
            x0_H = simde_mm_madd_epi16(x0_H, x0_H);

            sum = simde_mm_add_epi32(sum, simde_mm_add_epi32(x0_L, x0_H));

            coeffTemp += inputStride;
            reconCoeffTemp += reconStride;
        } while (--rowCount);

        input += 16;
        recon += 16;
        colCount -= 16;
    } while (colCount > 0);

    sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
    sum = simde_mm_unpacklo_epi32(sum, sum);
    sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
    spatialDistortion = simde_mm_extract_epi32(sum, 0);

    return spatialDistortion;

};
