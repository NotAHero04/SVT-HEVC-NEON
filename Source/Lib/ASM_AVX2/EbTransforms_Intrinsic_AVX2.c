/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbTransforms_AVX2.h"
#include "EbDefinitions.h"

#include "../../../simde/simde/x86/avx2.h"

// Coefficients for forward 16/32-point transform
#ifdef __GNUC__
__attribute__((aligned(16)))
#endif
EB_ALIGN(32) const EB_S16 EbHevcCoeff_tbl_AVX2[48 * 16] =
{
    64, 64, 89, 75, 83, 36, 75, -18, 64, 64, 89, 75, 83, 36, 75, -18, 64, -64, 50, -89, 36, -83, 18, -50, 64, -64, 50, -89, 36, -83, 18, -50,
    64, 64, 50, 18, -36, -83, -89, -50, 64, 64, 50, 18, -36, -83, -89, -50, -64, 64, 18, 75, 83, -36, 75, -89, -64, 64, 18, 75, 83, -36, 75, -89,
    64, 64, -18, -50, -83, -36, 50, 89, 64, 64, -18, -50, -83, -36, 50, 89, 64, -64, -75, -18, -36, 83, 89, -75, 64, -64, -75, -18, -36, 83, 89, -75,
    64, 64, -75, -89, 36, 83, 18, -75, 64, 64, -75, -89, 36, 83, 18, -75, -64, 64, 89, -50, -83, 36, 50, -18, -64, 64, 89, -50, -83, 36, 50, -18,
    90, 87, 87, 57, 80, 9, 70, -43, 90, 87, 87, 57, 80, 9, 70, -43, 57, -80, 43, -90, 25, -70, 9, -25, 57, -80, 43, -90, 25, -70, 9, -25,
    80, 70, 9, -43, -70, -87, -87, 9, 80, 70, 9, -43, -70, -87, -87, 9, -25, 90, 57, 25, 90, -80, 43, -57, -25, 90, 57, 25, 90, -80, 43, -57,
    57, 43, -80, -90, -25, 57, 90, 25, 57, 43, -80, -90, -25, 57, 90, 25, -9, -87, -87, 70, 43, 9, 70, -80, -9, -87, -87, 70, 43, 9, 70, -80,
    25, 9, -70, -25, 90, 43, -80, -57, 25, 9, -70, -25, 90, 43, -80, -57, 43, 70, 9, -80, -57, 87, 87, -90, 43, 70, 9, -80, -57, 87, 87, -90,
    90, 90, 90, 82, 88, 67, 85, 46, 90, 90, 90, 82, 88, 67, 85, 46, 82, 22, 78, -4, 73, -31, 67, -54, 82, 22, 78, -4, 73, -31, 67, -54,
    61, -73, 54, -85, 46, -90, 38, -88, 61, -73, 54, -85, 46, -90, 38, -88, 31, -78, 22, -61, 13, -38, 4, -13, 31, -78, 22, -61, 13, -38, 4, -13,
    88, 85, 67, 46, 31, -13, -13, -67, 88, 85, 67, 46, 31, -13, -13, -67, -54, -90, -82, -73, -90, -22, -78, 38, -54, -90, -82, -73, -90, -22, -78, 38,
    -46, 82, -4, 88, 38, 54, 73, -4, -46, 82, -4, 88, 38, 54, 73, -4, 90, -61, 85, -90, 61, -78, 22, -31, 90, -61, 85, -90, 61, -78, 22, -31,
    82, 78, 22, -4, -54, -82, -90, -73, 82, 78, 22, -4, -54, -82, -90, -73, -61, 13, 13, 85, 78, 67, 85, -22, -61, 13, 13, 85, 78, 67, 85, -22,
    31, -88, -46, -61, -90, 31, -67, 90, 31, -88, -46, -61, -90, 31, -67, 90, 4, 54, 73, -38, 88, -90, 38, -46, 4, 54, 73, -38, 88, -90, 38, -46,
    73, 67, -31, -54, -90, -78, -22, 38, 73, 67, -31, -54, -90, -78, -22, 38, 78, 85, 67, -22, -38, -90, -90, 4, 78, 85, 67, -22, -38, -90, -90, 4,
    -13, 90, 82, 13, 61, -88, -46, -31, -13, 90, 82, 13, 61, -88, -46, -31, -88, 82, -4, 46, 85, -73, 54, -61, -88, 82, -4, 46, 85, -73, 54, -61,
    61, 54, -73, -85, -46, -4, 82, 88, 61, 54, -73, -85, -46, -4, 82, 88, 31, -46, -88, -61, -13, 82, 90, 13, 31, -46, -88, -61, -13, 82, 90, 13,
    -4, -90, -90, 38, 22, 67, 85, -78, -4, -90, -90, 38, 22, 67, 85, -78, -38, -22, -78, 90, 54, -31, 67, -73, -38, -22, -78, 90, 54, -31, 67, -73,
    46, 38, -90, -88, 38, 73, 54, -4, 46, 38, -90, -88, 38, 73, 54, -4, -90, -67, 31, 90, 61, -46, -88, -31, -90, -67, 31, 90, 61, -46, -88, -31,
    22, 85, 67, -78, -85, 13, 13, 61, 22, 85, 67, -78, -85, 13, 13, 61, 73, -90, -82, 54, 4, 22, 78, -82, 73, -90, -82, 54, 4, 22, 78, -82,
    31, 22, -78, -61, 90, 85, -61, -90, 31, 22, -78, -61, 90, 85, -61, -90, 4, 73, 54, -38, -88, -4, 82, 46, 4, 73, 54, -38, -88, -4, 82, 46,
    -38, -78, -22, 90, 73, -82, -90, 54, -38, -78, -22, 90, 73, -82, -90, 54, 67, -13, -13, -31, -46, 67, 85, -88, 67, -13, -13, -31, -46, 67, 85, -88,
    13, 4, -38, -13, 61, 22, -78, -31, 13, 4, -38, -13, 61, 22, -78, -31, 88, 38, -90, -46, 85, 54, -73, -61, 88, 38, -90, -46, 85, 54, -73, -61,
    54, 67, -31, -73, 4, 78, 22, -82, 54, 67, -31, -73, 4, 78, 22, -82, -46, 85, 67, -88, -82, 90, 90, -90, -46, 85, 67, -88, -82, 90, 90, -90
};


/*******************************************************************************
* Requirement: areaSize = 4, 8, or areaSize % 16 = 0
*******************************************************************************/
void QuantizeInvQuantizeNxN_AVX2_INTRIN(
    EB_S16          *coeff,
    const EB_U32     coeffStride,
    EB_S16          *quantCoeff,
    EB_S16          *reconCoeff,
    const EB_U32     qFunc,
    const EB_U32     q_offset,
    const EB_S32     shiftedQBits,
    const EB_S32     shiftedFFunc,
    const EB_S32     iq_offset,
    const EB_S32     shiftNum,
    const EB_U32     areaSize,
    EB_U32          *nonzerocoeff)
{
    unsigned row, col;
    simde__m128i q = simde_mm_set1_epi16((EB_S16)qFunc);
    simde__m128i o = simde_mm_set1_epi32(q_offset);
    simde__m128i s = simde_mm_cvtsi32_si128(shiftedQBits);
    simde__m128i iq = simde_mm_set1_epi16((EB_S16)shiftedFFunc);
    simde__m128i io = simde_mm_set1_epi32(iq_offset);
    simde__m128i is = simde_mm_cvtsi32_si128(shiftNum);
    simde__m256i z = simde_mm256_setzero_si256();
    simde__m256i qq = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(q), q, 0x1);
    simde__m256i oo = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(o), o, 0x1);
    simde__m256i iqq = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(iq), iq, 0x1);
    simde__m256i ioo = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(io), io, 0x1);

    if (areaSize == 4) {
        simde__m128i a, b;
        simde__m256i a0, a1, b0, b1, x;
        simde__m256i y = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)coeff), simde_mm_loadl_epi64((simde__m128i*)(coeff + coeffStride)))),
            simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)(coeff + 2 * coeffStride)), simde_mm_loadl_epi64((simde__m128i*)(coeff + 3 * coeffStride))), 0x1);

        x = simde_mm256_abs_epi16(y);
        a0 = simde_mm256_mullo_epi16(x, qq);
        a1 = simde_mm256_mulhi_epi16(x, qq);
        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b0 = simde_mm256_add_epi32(b0, oo);
        b1 = simde_mm256_add_epi32(b1, oo);
        b0 = simde_mm256_sra_epi32(b0, s);
        b1 = simde_mm256_sra_epi32(b1, s);
        x = simde_mm256_packs_epi32(b0, b1);
        z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(x, simde_mm256_setzero_si256()));
        x = simde_mm256_sign_epi16(x, y);
        a = simde_mm256_extracti128_si256(x, 0);
        b = simde_mm256_extracti128_si256(x, 1);
        simde_mm_storel_epi64((simde__m128i *)quantCoeff, a);
        simde_mm_storel_epi64((simde__m128i *)(quantCoeff + coeffStride), simde_mm_srli_si128(a, 8));
        simde_mm_storel_epi64((simde__m128i *)(quantCoeff + 2 * coeffStride), b);
        simde_mm_storel_epi64((simde__m128i *)(quantCoeff + 3 * coeffStride), simde_mm_srli_si128(b, 8));

        a0 = simde_mm256_mullo_epi16(x, iqq);
        a1 = simde_mm256_mulhi_epi16(x, iqq);
        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b0 = simde_mm256_add_epi32(b0, ioo);
        b1 = simde_mm256_add_epi32(b1, ioo);
        b0 = simde_mm256_sra_epi32(b0, is);
        b1 = simde_mm256_sra_epi32(b1, is);
        x = simde_mm256_packs_epi32(b0, b1);
        a = simde_mm256_extracti128_si256(x, 0);
        b = simde_mm256_extracti128_si256(x, 1);
        simde_mm_storel_epi64((simde__m128i *)reconCoeff, a);
        simde_mm_storel_epi64((simde__m128i *)(reconCoeff + coeffStride), simde_mm_srli_si128(a, 8));
        simde_mm_storel_epi64((simde__m128i *)(reconCoeff + 2 * coeffStride), b);
        simde_mm_storel_epi64((simde__m128i *)(reconCoeff + 3 * coeffStride), simde_mm_srli_si128(b, 8));
    }
    else if (areaSize == 8) {
        row = 0;
        do {
            simde__m256i a0, a1, b0, b1, x;
            simde__m256i y = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*row))),
                simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*(row + 1))), 0x1);

            x = simde_mm256_abs_epi16(y);
            a0 = simde_mm256_mullo_epi16(x, qq);
            a1 = simde_mm256_mulhi_epi16(x, qq);
            b0 = simde_mm256_unpacklo_epi16(a0, a1);
            b1 = simde_mm256_unpackhi_epi16(a0, a1);
            b0 = simde_mm256_add_epi32(b0, oo);
            b1 = simde_mm256_add_epi32(b1, oo);
            b0 = simde_mm256_sra_epi32(b0, s);
            b1 = simde_mm256_sra_epi32(b1, s);
            x = simde_mm256_packs_epi32(b0, b1);
            z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(x, simde_mm256_setzero_si256()));
            x = simde_mm256_sign_epi16(x, y);
            simde_mm_storeu_si128((simde__m128i *)(quantCoeff + coeffStride*row), simde_mm256_extracti128_si256(x, 0));
            simde_mm_storeu_si128((simde__m128i *)(quantCoeff + coeffStride*(row + 1)), simde_mm256_extracti128_si256(x, 1));

            a0 = simde_mm256_mullo_epi16(x, iqq);
            a1 = simde_mm256_mulhi_epi16(x, iqq);
            b0 = simde_mm256_unpacklo_epi16(a0, a1);
            b1 = simde_mm256_unpackhi_epi16(a0, a1);
            b0 = simde_mm256_add_epi32(b0, ioo);
            b1 = simde_mm256_add_epi32(b1, ioo);
            b0 = simde_mm256_sra_epi32(b0, is);
            b1 = simde_mm256_sra_epi32(b1, is);
            x = simde_mm256_packs_epi32(b0, b1);
            simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*row), simde_mm256_extracti128_si256(x, 0));
            simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*(row + 1)), simde_mm256_extracti128_si256(x, 1));

            row += 2;
        } while (row < areaSize);
    }
    else {
        row = 0;
        do {
            col = 0;
            do {
                simde__m256i a0, a1, b0, b1, x;
                simde__m256i y = simde_mm256_loadu_si256((simde__m256i *)(coeff + coeffStride*row + col));

                x = simde_mm256_abs_epi16(y);
                a0 = simde_mm256_mullo_epi16(x, qq);
                a1 = simde_mm256_mulhi_epi16(x, qq);
                b0 = simde_mm256_unpacklo_epi16(a0, a1);
                b1 = simde_mm256_unpackhi_epi16(a0, a1);
                b0 = simde_mm256_add_epi32(b0, oo);
                b1 = simde_mm256_add_epi32(b1, oo);
                b0 = simde_mm256_sra_epi32(b0, s);
                b1 = simde_mm256_sra_epi32(b1, s);
                x = simde_mm256_packs_epi32(b0, b1);
                z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(x, simde_mm256_setzero_si256()));
                x = simde_mm256_sign_epi16(x, y);
                simde_mm256_storeu_si256((simde__m256i *)(quantCoeff + coeffStride*row + col), x);
				simde__m256i zer = simde_mm256_setzero_si256();
				simde__m256i cmp = simde_mm256_cmpeq_epi16(x, zer);
				int msk = simde_mm256_movemask_epi8(cmp);

				if ((EB_U32)msk != 0xFFFFFFFF)
				{
                a0 = simde_mm256_mullo_epi16(x, iqq);
                a1 = simde_mm256_mulhi_epi16(x, iqq);
                b0 = simde_mm256_unpacklo_epi16(a0, a1);
                b1 = simde_mm256_unpackhi_epi16(a0, a1);
                b0 = simde_mm256_add_epi32(b0, ioo);
                b1 = simde_mm256_add_epi32(b1, ioo);
                b0 = simde_mm256_sra_epi32(b0, is);
                b1 = simde_mm256_sra_epi32(b1, is);
                x = simde_mm256_packs_epi32(b0, b1);
                simde_mm256_storeu_si256((simde__m256i *)(reconCoeff + coeffStride*row + col), x);
				}
				else{
					simde_mm256_storeu_si256((simde__m256i *)(reconCoeff + coeffStride*row + col), zer);
				}
                col += 16;
            } while (col < areaSize);

            row++;
        } while (row < areaSize);
    }

    z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 7));
    *nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));

}

void QuantizeInvQuantize8x8_AVX2_INTRIN(
    EB_S16          *coeff,
    const EB_U32     coeffStride,
    EB_S16          *quantCoeff,
    EB_S16          *reconCoeff,
    const EB_U32     qFunc,
    const EB_U32     q_offset,
    const EB_S32     shiftedQBits,
    const EB_S32     shiftedFFunc,
    const EB_S32     iq_offset,
    const EB_S32     shiftNum,
    const EB_U32     areaSize,
    EB_U32          *nonzerocoeff)
{
    unsigned row;
    simde__m128i q = simde_mm_set1_epi16((EB_S16)qFunc);
    simde__m128i o = simde_mm_set1_epi32(q_offset);
    simde__m128i s = simde_mm_cvtsi32_si128(shiftedQBits);
    simde__m128i iq = simde_mm_set1_epi16((EB_S16)shiftedFFunc);
    simde__m128i io = simde_mm_set1_epi32(iq_offset);
    simde__m128i is = simde_mm_cvtsi32_si128(shiftNum);
    simde__m256i z = simde_mm256_setzero_si256();
    simde__m256i qq = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(q), q, 0x1);
    simde__m256i oo = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(o), o, 0x1);
    simde__m256i iqq = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(iq), iq, 0x1);
    simde__m256i ioo = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(io), io, 0x1);

    (void)areaSize;

    row = 0;
    do {
        simde__m256i a0, a1, b0, b1, x;
        simde__m256i y = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*row))), simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*(row + 1))), 0x1);

        x = simde_mm256_abs_epi16(y);
        a0 = simde_mm256_mullo_epi16(x, qq);
        a1 = simde_mm256_mulhi_epi16(x, qq);
        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b0 = simde_mm256_add_epi32(b0, oo);
        b1 = simde_mm256_add_epi32(b1, oo);
        b0 = simde_mm256_sra_epi32(b0, s);
        b1 = simde_mm256_sra_epi32(b1, s);
        x = simde_mm256_packs_epi32(b0, b1);
        z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(x, simde_mm256_setzero_si256()));
        x = simde_mm256_sign_epi16(x, y);
        simde_mm_storeu_si128((simde__m128i *)(quantCoeff + coeffStride*row), simde_mm256_extracti128_si256(x, 0));
        simde_mm_storeu_si128((simde__m128i *)(quantCoeff + coeffStride*(row + 1)), simde_mm256_extracti128_si256(x, 1));
		simde__m256i zer = simde_mm256_setzero_si256();
		simde__m256i cmp = simde_mm256_cmpeq_epi16(x, zer);
		int msk = simde_mm256_movemask_epi8(cmp);

		if ((EB_U32)msk != 0xFFFFFFFF)
		{
        a0 = simde_mm256_mullo_epi16(x, iqq);
        a1 = simde_mm256_mulhi_epi16(x, iqq);
        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b0 = simde_mm256_add_epi32(b0, ioo);
        b1 = simde_mm256_add_epi32(b1, ioo);
        b0 = simde_mm256_sra_epi32(b0, is);
        b1 = simde_mm256_sra_epi32(b1, is);
        x = simde_mm256_packs_epi32(b0, b1);
        simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*row), simde_mm256_extracti128_si256(x, 0));
        simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*(row + 1)), simde_mm256_extracti128_si256(x, 1));
		}
		else{
			simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*row), simde_mm_setzero_si128());
			simde_mm_storeu_si128((simde__m128i *)(reconCoeff + coeffStride*(row + 1)), simde_mm_setzero_si128());
		}
        row += 2;
    } while (row < 8);

    z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 7));
    *nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));
}


// transpose 16x16 block of data
void EbHevcTranspose16_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride)
{
    EB_U32 i;
    for (i = 0; i < 2; i++)
    {
        simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
        simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

        a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 0)*src_stride));
        a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 1)*src_stride));
        a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 2)*src_stride));
        a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 3)*src_stride));
        a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 4)*src_stride));
        a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 5)*src_stride));
        a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 6)*src_stride));
        a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 7)*src_stride));

        b0 = simde_mm256_unpacklo_epi16(a0, a4);
        b1 = simde_mm256_unpacklo_epi16(a1, a5);
        b2 = simde_mm256_unpacklo_epi16(a2, a6);
        b3 = simde_mm256_unpacklo_epi16(a3, a7);
        b4 = simde_mm256_unpackhi_epi16(a0, a4);
        b5 = simde_mm256_unpackhi_epi16(a1, a5);
        b6 = simde_mm256_unpackhi_epi16(a2, a6);
        b7 = simde_mm256_unpackhi_epi16(a3, a7);

        a0 = simde_mm256_unpacklo_epi16(b0, b2);
        a1 = simde_mm256_unpacklo_epi16(b1, b3);
        a2 = simde_mm256_unpackhi_epi16(b0, b2);
        a3 = simde_mm256_unpackhi_epi16(b1, b3);
        a4 = simde_mm256_unpacklo_epi16(b4, b6);
        a5 = simde_mm256_unpacklo_epi16(b5, b7);
        a6 = simde_mm256_unpackhi_epi16(b4, b6);
        a7 = simde_mm256_unpackhi_epi16(b5, b7);

        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b2 = simde_mm256_unpacklo_epi16(a2, a3);
        b3 = simde_mm256_unpackhi_epi16(a2, a3);
        b4 = simde_mm256_unpacklo_epi16(a4, a5);
        b5 = simde_mm256_unpackhi_epi16(a4, a5);
        b6 = simde_mm256_unpacklo_epi16(a6, a7);
        b7 = simde_mm256_unpackhi_epi16(a6, a7);

        simde_mm_storeu_si128((simde__m128i *)(dst + 0 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 1 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 2 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 3 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 4 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 5 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 6 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 7 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 8 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 9 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 10 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 11 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 12 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 13 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 14 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 15 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 1));
    }
}

// transpose 32x32 block of data
void EbHevcTranspose32_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride)
{
    EB_U32 i, j;
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 2; j++)
        {
            simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
            simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 0)*src_stride + 16 * j));
            a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 1)*src_stride + 16 * j));
            a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 2)*src_stride + 16 * j));
            a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 3)*src_stride + 16 * j));
            a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 4)*src_stride + 16 * j));
            a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 5)*src_stride + 16 * j));
            a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 6)*src_stride + 16 * j));
            a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 7)*src_stride + 16 * j));

            b0 = simde_mm256_unpacklo_epi16(a0, a4);
            b1 = simde_mm256_unpacklo_epi16(a1, a5);
            b2 = simde_mm256_unpacklo_epi16(a2, a6);
            b3 = simde_mm256_unpacklo_epi16(a3, a7);
            b4 = simde_mm256_unpackhi_epi16(a0, a4);
            b5 = simde_mm256_unpackhi_epi16(a1, a5);
            b6 = simde_mm256_unpackhi_epi16(a2, a6);
            b7 = simde_mm256_unpackhi_epi16(a3, a7);

            a0 = simde_mm256_unpacklo_epi16(b0, b2);
            a1 = simde_mm256_unpacklo_epi16(b1, b3);
            a2 = simde_mm256_unpackhi_epi16(b0, b2);
            a3 = simde_mm256_unpackhi_epi16(b1, b3);
            a4 = simde_mm256_unpacklo_epi16(b4, b6);
            a5 = simde_mm256_unpacklo_epi16(b5, b7);
            a6 = simde_mm256_unpackhi_epi16(b4, b6);
            a7 = simde_mm256_unpackhi_epi16(b5, b7);

            b0 = simde_mm256_unpacklo_epi16(a0, a1);
            b1 = simde_mm256_unpackhi_epi16(a0, a1);
            b2 = simde_mm256_unpacklo_epi16(a2, a3);
            b3 = simde_mm256_unpackhi_epi16(a2, a3);
            b4 = simde_mm256_unpacklo_epi16(a4, a5);
            b5 = simde_mm256_unpackhi_epi16(a4, a5);
            b6 = simde_mm256_unpacklo_epi16(a6, a7);
            b7 = simde_mm256_unpackhi_epi16(a6, a7);

            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 0)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 1)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 2)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 3)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 4)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 5)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 6)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 7)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 0));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 8)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 9)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 10)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 11)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 12)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 13)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 14)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 1));
            simde_mm_storeu_si128((simde__m128i *)(dst + (16 * j + 15)*dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 1));
        }
    }
}

void PfreqTranspose32_AVX2_INTRIN(
    EB_S16 *src,
    EB_U32  src_stride,
    EB_S16 *dst,
    EB_U32  dst_stride)
{
    EB_U32 i;
    for (i = 0; i < 4; i++)
    {
        simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
        simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

        a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 0)*src_stride));
        a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 1)*src_stride));
        a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 2)*src_stride));
        a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 3)*src_stride));
        a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 4)*src_stride));
        a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 5)*src_stride));
        a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 6)*src_stride));
        a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 7)*src_stride));

        b0 = simde_mm256_unpacklo_epi16(a0, a4);
        b1 = simde_mm256_unpacklo_epi16(a1, a5);
        b2 = simde_mm256_unpacklo_epi16(a2, a6);
        b3 = simde_mm256_unpacklo_epi16(a3, a7);
        b4 = simde_mm256_unpackhi_epi16(a0, a4);
        b5 = simde_mm256_unpackhi_epi16(a1, a5);
        b6 = simde_mm256_unpackhi_epi16(a2, a6);
        b7 = simde_mm256_unpackhi_epi16(a3, a7);

        a0 = simde_mm256_unpacklo_epi16(b0, b2);
        a1 = simde_mm256_unpacklo_epi16(b1, b3);
        a2 = simde_mm256_unpackhi_epi16(b0, b2);
        a3 = simde_mm256_unpackhi_epi16(b1, b3);
        a4 = simde_mm256_unpacklo_epi16(b4, b6);
        a5 = simde_mm256_unpacklo_epi16(b5, b7);
        a6 = simde_mm256_unpackhi_epi16(b4, b6);
        a7 = simde_mm256_unpackhi_epi16(b5, b7);

        b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);
        b2 = simde_mm256_unpacklo_epi16(a2, a3);
        b3 = simde_mm256_unpackhi_epi16(a2, a3);
        b4 = simde_mm256_unpacklo_epi16(a4, a5);
        b5 = simde_mm256_unpackhi_epi16(a4, a5);
        b6 = simde_mm256_unpacklo_epi16(a6, a7);
        b7 = simde_mm256_unpackhi_epi16(a6, a7);

        simde_mm_storeu_si128((simde__m128i *)(dst + 0 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 1 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 2 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 3 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 4 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 5 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 6 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 7 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 0));
        simde_mm_storeu_si128((simde__m128i *)(dst + 8 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 9 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 10 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 11 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 12 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 13 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 14 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 1));
        simde_mm_storeu_si128((simde__m128i *)(dst + 15 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 1));
    }
}

void PfreqSecTranspose32_AVX2_INTRIN(
	EB_S16 *src,
	EB_U32  src_stride,
	EB_S16 *dst,
	EB_U32  dst_stride)
{
	EB_U32 i;
	//  for (i = 0; i < 4; i++)
	for (i = 0; i < 2; i++)
	{
		simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
		simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

		a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 0)*src_stride));
		a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 1)*src_stride));
		a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 2)*src_stride));
		a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 3)*src_stride));
		a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 4)*src_stride));
		a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 5)*src_stride));
		a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 6)*src_stride));
		a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 7)*src_stride));

		b0 = simde_mm256_unpacklo_epi16(a0, a4);
		b1 = simde_mm256_unpacklo_epi16(a1, a5);
		b2 = simde_mm256_unpacklo_epi16(a2, a6);
		b3 = simde_mm256_unpacklo_epi16(a3, a7);
		b4 = simde_mm256_unpackhi_epi16(a0, a4);
		b5 = simde_mm256_unpackhi_epi16(a1, a5);
		b6 = simde_mm256_unpackhi_epi16(a2, a6);
		b7 = simde_mm256_unpackhi_epi16(a3, a7);

		a0 = simde_mm256_unpacklo_epi16(b0, b2);
		a1 = simde_mm256_unpacklo_epi16(b1, b3);
		a2 = simde_mm256_unpackhi_epi16(b0, b2);
		a3 = simde_mm256_unpackhi_epi16(b1, b3);
		a4 = simde_mm256_unpacklo_epi16(b4, b6);
		a5 = simde_mm256_unpacklo_epi16(b5, b7);
		a6 = simde_mm256_unpackhi_epi16(b4, b6);
		a7 = simde_mm256_unpackhi_epi16(b5, b7);

		b0 = simde_mm256_unpacklo_epi16(a0, a1);
		b1 = simde_mm256_unpackhi_epi16(a0, a1);
		b2 = simde_mm256_unpacklo_epi16(a2, a3);
		b3 = simde_mm256_unpackhi_epi16(a2, a3);
		b4 = simde_mm256_unpacklo_epi16(a4, a5);
		b5 = simde_mm256_unpackhi_epi16(a4, a5);
		b6 = simde_mm256_unpacklo_epi16(a6, a7);
		b7 = simde_mm256_unpackhi_epi16(a6, a7);

		simde_mm_storeu_si128((simde__m128i *)(dst + 0 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 1 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 2 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 3 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 4 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 5 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 6 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 7 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 8 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 9 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 10 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 11 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 12 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 13 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 14 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 15 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 1));
	}
}

// 16-point forward transform (16 rows)
void transform16_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride, EB_S16 shift)
{
    EB_U32 i;
    simde__m128i s0 = simde_mm_cvtsi32_si128(shift);
    simde__m256i o0 = simde_mm256_set1_epi32(1 << (shift - 1));
    const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

    for (i = 0; i < 16; i += 2)
    {
        simde__m256i x0, x1;
        simde__m256i y0, y1;
        simde__m256i a0, a1, a2, a3;
        simde__m256i b0, b1, b2, b3;
        //simde_mm_loadu_si128((simde__m128i*)(ref + refStride))
        y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x00)), 0x1);
        y1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x08)), 0x1);

        // 16-point butterfly
        y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        x0 = simde_mm256_add_epi16(y0, y1);
        x1 = simde_mm256_sub_epi16(y0, y1);

        a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

        a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

        a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

        b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
        b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
        b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
        b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);

        x0 = simde_mm256_packs_epi32(b0, b1);
        x1 = simde_mm256_packs_epi32(b2, b3);

        y0 = simde_mm256_unpacklo_epi16(x0, x1);
        y1 = simde_mm256_unpackhi_epi16(x0, x1);
        x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(y0, 0)), simde_mm256_extracti128_si256(y1, 0), 0x1);
        x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(y0, 1)), simde_mm256_extracti128_si256(y1, 1), 0x1);

        simde_mm256_storeu_si256((simde__m256i *)(dst + i*dst_stride), x0);
        simde_mm256_storeu_si256((simde__m256i *)(dst + (i + 1)*dst_stride), x1);
    }
}

// 32-point forward transform (32 rows)
void EbHevcTransform32_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride, EB_U32 shift)
{
    EB_U32 i;
    simde__m128i s0;
    simde__m256i o0;
    const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

    shift &= 0x0000FFFF; // Redundant code to fix Visual Studio 2012 AVX2 compiler error
    s0 = simde_mm_cvtsi32_si128(shift);
    o0 = simde_mm256_set1_epi32(1 << (shift - 1));

    for (i = 0; i < 16; i++)
    {
        simde__m256i x0, x1, x2, x3;
        simde__m256i y0, y1, y2, y3;
        simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
        simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

        x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + src_stride + 0x00)), 0x1);
        x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + src_stride + 0x08)), 0x1);
        x2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + 0x10))), simde_mm_loadu_si128((const simde__m128i *)(src + src_stride + 0x10)), 0x1);
        x3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + 0x18))), simde_mm_loadu_si128((const simde__m128i *)(src + src_stride + 0x18)), 0x1);

        // 32-point butterfly
        x2 = simde_mm256_shuffle_epi8(x2, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
        x3 = simde_mm256_shuffle_epi8(x3, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        y0 = simde_mm256_add_epi16(x0, x3);
        y1 = simde_mm256_add_epi16(x1, x2);

        y2 = simde_mm256_sub_epi16(x0, x3);
        y3 = simde_mm256_sub_epi16(x1, x2);

        // 16-point butterfly
        y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        x0 = simde_mm256_add_epi16(y0, y1);
        x1 = simde_mm256_sub_epi16(y0, y1);

        x2 = y2;
        x3 = y3;

        a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

        a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
        a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

        a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
        a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

        a4 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[16]);
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[20]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[24]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[28]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[32]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[36]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[40]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[44]));

        a5 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[17]);
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[21]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[25]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[29]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[33]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[37]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[41]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[45]));

        a6 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[18]);
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[22]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[26]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[30]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[34]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[38]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[42]));
        a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[46]));

        a7 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[19]);
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[23]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[27]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[31]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[35]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[39]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[43]));
        a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[47]));

        b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
        b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
        b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
        b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);
        b4 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a4, o0), s0);
        b5 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a5, o0), s0);
        b6 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a6, o0), s0);
        b7 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a7, o0), s0);

        x0 = simde_mm256_packs_epi32(b0, b1);
        x1 = simde_mm256_packs_epi32(b2, b3);
        x2 = simde_mm256_packs_epi32(b4, b5);
        x3 = simde_mm256_packs_epi32(b6, b7);

        y0 = simde_mm256_unpacklo_epi16(x0, x1);
        y1 = simde_mm256_unpackhi_epi16(x0, x1);
        y2 = x2;
        y3 = x3;
        x0 = simde_mm256_unpacklo_epi16(y0, y2);
        x1 = simde_mm256_unpackhi_epi16(y0, y2);
        x2 = simde_mm256_unpacklo_epi16(y1, y3);
        x3 = simde_mm256_unpackhi_epi16(y1, y3);

        y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 0)), simde_mm256_extracti128_si256(x1, 0), 0x1);
        y1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x2, 0)), simde_mm256_extracti128_si256(x3, 0), 0x1);
        y2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 1)), simde_mm256_extracti128_si256(x1, 1), 0x1);
        y3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x2, 1)), simde_mm256_extracti128_si256(x3, 1), 0x1);
        simde_mm256_storeu_si256((simde__m256i *)(dst + 0x00), y0);
        simde_mm256_storeu_si256((simde__m256i *)(dst + 0x10), y1);
        simde_mm256_storeu_si256((simde__m256i *)(dst + dst_stride + 0x00), y2);
        simde_mm256_storeu_si256((simde__m256i *)(dst + dst_stride + 0x10), y3);

        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

void Pfreq1DTransform32_AVX2_INTRIN(
    EB_S16 *src,
    EB_U32  src_stride,
    EB_S16 *dst,
    EB_U32  dst_stride,
    EB_U32  shift)
{
    EB_U32 i;
    simde__m128i s0 = simde_mm_cvtsi32_si128(shift);
    simde__m256i o0 = simde_mm256_set1_epi32(1 << (shift - 1));
    const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

    for (i = 0; i < 32; i += 2)
    {
        simde__m256i x0, x1, x2, x3;
        simde__m256i y0, y1, y2, y3;
        simde__m256i a0, a2, a4, a5;
        simde__m256i b0, b2, b4, b5;

        //b1=s0;
        //b3=s0;

        x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x00)), 0x1);
        x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x08)), 0x1);
        x2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x10))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x10)), 0x1);
        x3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x18))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x18)), 0x1);

        // 32-point butterfly
        x2 = simde_mm256_shuffle_epi8(x2, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
        x3 = simde_mm256_shuffle_epi8(x3, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        y0 = simde_mm256_add_epi16(x0, x3);
        y1 = simde_mm256_add_epi16(x1, x2);

        y2 = simde_mm256_sub_epi16(x0, x3);
        y3 = simde_mm256_sub_epi16(x1, x2);

        // 16-point butterfly
        y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        x0 = simde_mm256_add_epi16(y0, y1);
        x1 = simde_mm256_sub_epi16(y0, y1);

        x2 = y2;
        x3 = y3;

        a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

        //a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

        //a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

        a4 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[16]);
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[20]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[24]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[28]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[32]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[36]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[40]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[44]));

        a5 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[17]);
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[21]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[25]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[29]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[33]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[37]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[41]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[45]));

        //a6 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[18]);
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[22]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[26]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[30]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[34]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[38]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[42]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[46]));
        //
        //a7 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[19]);
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[23]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[27]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[31]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[35]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[39]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[43]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[47]));

        b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
        //b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
        b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
        //b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);
        b4 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a4, o0), s0);
        b5 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a5, o0), s0);
        //b6 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a6, o0), s0);
        //b7 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a7, o0), s0);

        x0 = simde_mm256_packs_epi32(b0, b0);
        x1 = simde_mm256_packs_epi32(b2, b2);
        x2 = simde_mm256_packs_epi32(b4, b5);
        //x3 = simde_mm256_packs_epi32(b6, b7);

        y0 = simde_mm256_unpacklo_epi16(x0, x1);
        //y1 = simde_mm256_unpackhi_epi16(x0, x1);
        y2 = x2;
        //y3 = x3;
        x0 = simde_mm256_unpacklo_epi16(y0, y2);
        x1 = simde_mm256_unpackhi_epi16(y0, y2);
        //x2 = simde_mm256_unpacklo_epi16(y1, y3);
        //x3 = simde_mm256_unpackhi_epi16(y1, y3);

        y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 0)), simde_mm256_extracti128_si256(x1, 0), 0x1);
        y2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 1)), simde_mm256_extracti128_si256(x1, 1), 0x1);
        simde_mm256_storeu_si256((simde__m256i *)(dst + i*dst_stride), y0);
        simde_mm256_storeu_si256((simde__m256i *)(dst + (i + 1)*dst_stride), y2);
        //simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x10), x2);
        //simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x18), x3);
    }
}

void Pfreq2DTransform32_AVX2_INTRIN(
    EB_S16 *src,
    EB_U32  src_stride,
    EB_S16 *dst,
    EB_U32  dst_stride,
    EB_U32  shift)
{
    EB_U32 i;
    simde__m128i s0 = simde_mm_cvtsi32_si128(shift);
    simde__m256i o0 = simde_mm256_set1_epi32(1 << (shift - 1));
    const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

    for (i = 0; i < 16; i += 2)
    {
        simde__m256i x0, x1, x2, x3;
        simde__m256i y0, y1, y2, y3;
        simde__m256i a0, a2, a4, a5;
        simde__m256i b0, b2, b4, b5;

        //b1=s0;
        //b3=s0;

        x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x00)), 0x1);
        x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x08)), 0x1);
        x2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x10))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x10)), 0x1);
        x3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x18))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x18)), 0x1);

        // 32-point butterfly
        x2 = simde_mm256_shuffle_epi8(x2, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
        x3 = simde_mm256_shuffle_epi8(x3, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        y0 = simde_mm256_add_epi16(x0, x3);
        y1 = simde_mm256_add_epi16(x1, x2);

        y2 = simde_mm256_sub_epi16(x0, x3);
        y3 = simde_mm256_sub_epi16(x1, x2);

        // 16-point butterfly
        y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        x0 = simde_mm256_add_epi16(y0, y1);
        x1 = simde_mm256_sub_epi16(y0, y1);

        x2 = y2;
        x3 = y3;

        a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

        //a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
        //a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

        //a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
        //a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

        a4 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[16]);
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[20]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[24]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[28]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[32]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[36]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[40]));
        a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[44]));

        a5 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[17]);
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[21]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[25]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[29]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[33]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[37]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[41]));
        a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[45]));

        //a6 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[18]);
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[22]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[26]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[30]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[34]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[38]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[42]));
        //a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[46]));
        //
        //a7 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[19]);
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[23]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[27]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[31]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[35]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[39]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[43]));
        //a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[47]));

        b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
        //b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
        b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
        //b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);
        b4 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a4, o0), s0);
        b5 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a5, o0), s0);
        //b6 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a6, o0), s0);
        //b7 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a7, o0), s0);

        x0 = simde_mm256_packs_epi32(b0, b0);
        x1 = simde_mm256_packs_epi32(b2, b2);
        x2 = simde_mm256_packs_epi32(b4, b5);
        //x3 = simde_mm256_packs_epi32(b6, b7);

        y0 = simde_mm256_unpacklo_epi16(x0, x1);
        //y1 = simde_mm256_unpackhi_epi16(x0, x1);
        y2 = x2;
        //y3 = x3;
        x0 = simde_mm256_unpacklo_epi16(y0, y2);
        x1 = simde_mm256_unpackhi_epi16(y0, y2);
        //x2 = simde_mm256_unpacklo_epi16(y1, y3);
        //x3 = simde_mm256_unpackhi_epi16(y1, y3);

        y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 0)), simde_mm256_extracti128_si256(x1, 0), 0x1);
        y2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 1)), simde_mm256_extracti128_si256(x1, 1), 0x1);
        simde_mm256_storeu_si256((simde__m256i *)(dst + i*dst_stride), y0);
        simde_mm256_storeu_si256((simde__m256i *)(dst + (i + 1)*dst_stride), y2);
        //simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x10), x2);
        //simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x18), x3);
    }
}

void PfreqTransform32x32_AVX2_INTRIN(
    EB_S16 *src,
    const EB_U32 src_stride,
    EB_S16 *dst,
    const EB_U32 dst_stride,
    EB_S16 *intermediate,
    EB_U32 addshift)
{
    Pfreq1DTransform32_AVX2_INTRIN(src, src_stride, intermediate, 32, 6 + addshift);
    PfreqTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);
    Pfreq2DTransform32_AVX2_INTRIN(dst, dst_stride, intermediate, 32, 9);
	PfreqSecTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);
    return;
}

void PfreqN41DTransform32_AVX2_INTRIN(
	EB_S16 *src,
	EB_U32  src_stride,
	EB_S16 *dst,
	EB_U32  dst_stride,
	EB_U32  shift)
{
	EB_U32 i;
	simde__m128i s0 = simde_mm_cvtsi32_si128(shift);
	simde__m256i o0 = simde_mm256_set1_epi32(1 << (shift - 1));
	const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

	for (i = 0; i < 32; i += 2)
	{
		simde__m256i x0, x1, x2, x3;
		simde__m256i y0, y1, y2, y3;
		simde__m256i a0, a2, a4, a5;
		simde__m256i b0, b2, b4, b5;

		//b1=s0;
		//b3=s0;

		x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x00)), 0x1);
		x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x08)), 0x1);
		x2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x10))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x10)), 0x1);
		x3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x18))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x18)), 0x1);

		// 32-point butterfly
		x2 = simde_mm256_shuffle_epi8(x2, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
		x3 = simde_mm256_shuffle_epi8(x3, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

		y0 = simde_mm256_add_epi16(x0, x3);
		y1 = simde_mm256_add_epi16(x1, x2);

		y2 = simde_mm256_sub_epi16(x0, x3);
		y3 = simde_mm256_sub_epi16(x1, x2);

		// 16-point butterfly
		y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

		x0 = simde_mm256_add_epi16(y0, y1);
		x1 = simde_mm256_sub_epi16(y0, y1);

		x2 = y2;
		x3 = y3;

		a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

		//a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

		a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

		//a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

		a4 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[16]);
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[20]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[24]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[28]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[32]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[36]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[40]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[44]));

		a5 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[17]);
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[21]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[25]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[29]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[33]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[37]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[41]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[45]));

		//a6 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[18]);
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[22]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[26]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[30]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[34]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[38]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[42]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[46]));
		//
		//a7 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[19]);
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[23]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[27]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[31]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[35]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[39]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[43]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[47]));

		b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
		//b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
		b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
		//b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);
		b4 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a4, o0), s0);
		b5 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a5, o0), s0);
		//b6 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a6, o0), s0);
		//b7 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a7, o0), s0);

		x0 = simde_mm256_packs_epi32(b0, b0);
		x1 = simde_mm256_packs_epi32(b2, b2);
		x2 = simde_mm256_packs_epi32(b4, b5);
		//x3 = simde_mm256_packs_epi32(b6, b7);

		y0 = simde_mm256_unpacklo_epi16(x0, x1);
		//y1 = simde_mm256_unpackhi_epi16(x0, x1);
		y2 = x2;
		//y3 = x3;
		x0 = simde_mm256_unpacklo_epi16(y0, y2);
		//--//        x1 = simde_mm256_unpackhi_epi16(y0, y2);
		//x2 = simde_mm256_unpacklo_epi16(y1, y3);
		//x3 = simde_mm256_unpackhi_epi16(y1, y3);

		//--//  y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 0)), simde_mm256_extracti128_si256(x1, 0), 0x1);
		//--//  y2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 1)), simde_mm256_extracti128_si256(x1, 1), 0x1);
		//--//  simde_mm256_storeu_si256((simde__m256i *)(dst + i*dst_stride), y0);
		//--//  simde_mm256_storeu_si256((simde__m256i *)(dst + (i + 1)*dst_stride), y2);

		simde_mm_storeu_si128((simde__m128i *)(dst + i*dst_stride), simde_mm256_extracti128_si256(x0, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + (i + 1)*dst_stride), simde_mm256_extracti128_si256(x0, 1));

		//simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x10), x2);
		//simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x18), x3);
	}
}
void PfreqN42DTransform32_AVX2_INTRIN(
	EB_S16 *src,
	EB_U32  src_stride,
	EB_S16 *dst,
	EB_U32  dst_stride,
	EB_U32  shift)
{
	EB_U32 i;
	simde__m128i s0 = simde_mm_cvtsi32_si128(shift);
	simde__m256i o0 = simde_mm256_set1_epi32(1 << (shift - 1));
	const simde__m256i *coeff32 = (const simde__m256i *)EbHevcCoeff_tbl_AVX2;

	//for (i = 0; i < 16; i += 2)
	for (i = 0; i < 8; i += 2)
	{
		simde__m256i x0, x1, x2, x3;
		simde__m256i y0, y1, y2, y3;
		simde__m256i a0, a2, a4, a5;
		simde__m256i b0, b2, b4, b5;

		//b1=s0;
		//b3=s0;

		x0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x00))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x00)), 0x1);
		x1 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x08))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x08)), 0x1);
		x2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x10))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x10)), 0x1);
		x3 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)(src + i*src_stride + 0x18))), simde_mm_loadu_si128((const simde__m128i *)(src + (i + 1)*src_stride + 0x18)), 0x1);

		// 32-point butterfly
		x2 = simde_mm256_shuffle_epi8(x2, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
		x3 = simde_mm256_shuffle_epi8(x3, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

		y0 = simde_mm256_add_epi16(x0, x3);
		y1 = simde_mm256_add_epi16(x1, x2);

		y2 = simde_mm256_sub_epi16(x0, x3);
		y3 = simde_mm256_sub_epi16(x1, x2);

		// 16-point butterfly
		y1 = simde_mm256_shuffle_epi8(y1, simde_mm256_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

		x0 = simde_mm256_add_epi16(y0, y1);
		x1 = simde_mm256_sub_epi16(y0, y1);

		x2 = y2;
		x3 = y3;

		a0 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[0]);
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[2]));
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[4]));
		a0 = simde_mm256_add_epi32(a0, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[6]));

		//a1 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x00), coeff32[1]);
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0x55), coeff32[3]));
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xaa), coeff32[5]));
		//a1 = simde_mm256_add_epi32(a1, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x0, 0xff), coeff32[7]));

		a2 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[8]);
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[10]));
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[12]));
		a2 = simde_mm256_add_epi32(a2, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[14]));

		//a3 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x00), coeff32[9]);
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0x55), coeff32[11]));
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xaa), coeff32[13]));
		//a3 = simde_mm256_add_epi32(a3, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x1, 0xff), coeff32[15]));

		a4 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[16]);
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[20]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[24]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[28]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[32]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[36]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[40]));
		a4 = simde_mm256_add_epi32(a4, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[44]));

		a5 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[17]);
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[21]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[25]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[29]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[33]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[37]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[41]));
		a5 = simde_mm256_add_epi32(a5, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[45]));

		//a6 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[18]);
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[22]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[26]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[30]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[34]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[38]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[42]));
		//a6 = simde_mm256_add_epi32(a6, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[46]));
		//
		//a7 = simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x00), coeff32[19]);
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0x55), coeff32[23]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xaa), coeff32[27]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x2, 0xff), coeff32[31]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x00), coeff32[35]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0x55), coeff32[39]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xaa), coeff32[43]));
		//a7 = simde_mm256_add_epi32(a7, simde_mm256_madd_epi16(simde_mm256_shuffle_epi32(x3, 0xff), coeff32[47]));

		b0 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a0, o0), s0);
		//b1 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a1, o0), s0);
		b2 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a2, o0), s0);
		//b3 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a3, o0), s0);
		b4 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a4, o0), s0);
		b5 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a5, o0), s0);
		//b6 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a6, o0), s0);
		//b7 = simde_mm256_sra_epi32(simde_mm256_add_epi32(a7, o0), s0);

		x0 = simde_mm256_packs_epi32(b0, b0);
		x1 = simde_mm256_packs_epi32(b2, b2);
		x2 = simde_mm256_packs_epi32(b4, b5);
		//x3 = simde_mm256_packs_epi32(b6, b7);

		y0 = simde_mm256_unpacklo_epi16(x0, x1);
		//y1 = simde_mm256_unpackhi_epi16(x0, x1);
		y2 = x2;
		//y3 = x3;
		x0 = simde_mm256_unpacklo_epi16(y0, y2);
		//---//  x1 = simde_mm256_unpackhi_epi16(y0, y2);
		//x2 = simde_mm256_unpacklo_epi16(y1, y3);
		//x3 = simde_mm256_unpackhi_epi16(y1, y3);

		//---// y0 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 0)), simde_mm256_extracti128_si256(x1, 0), 0x1);
		//---//y2 = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(x0, 1)), simde_mm256_extracti128_si256(x1, 1), 0x1);

		//---// simde_mm256_storeu_si256((simde__m256i *)(dst + i*dst_stride), y0);
		//---// simde_mm256_storeu_si256((simde__m256i *)(dst + (i + 1)*dst_stride), y2);

		simde_mm_storeu_si128((simde__m128i *)(dst + i*dst_stride), simde_mm256_extracti128_si256(x0, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + (i + 1)*dst_stride), simde_mm256_extracti128_si256(x0, 1));


		//simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x10), x2);
		//simde_mm256_storeu_si256((simde__m256i *)(dst+i*dst_stride+0x18), x3);
	}
}
void PfreqN4FirstTranspose32_AVX2_INTRIN(
	EB_S16 *src,
	EB_U32  src_stride,
	EB_S16 *dst,
	EB_U32  dst_stride)
{
	EB_U32 i;
	for (i = 0; i < 4; i++)
	{
		simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
		simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

		a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 0)*src_stride));
		a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 1)*src_stride));
		a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 2)*src_stride));
		a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 3)*src_stride));
		a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 4)*src_stride));
		a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 5)*src_stride));
		a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 6)*src_stride));
		a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (8 * i + 7)*src_stride));

		b0 = simde_mm256_unpacklo_epi16(a0, a4);
		b1 = simde_mm256_unpacklo_epi16(a1, a5);
		b2 = simde_mm256_unpacklo_epi16(a2, a6);
		b3 = simde_mm256_unpacklo_epi16(a3, a7);
		b4 = simde_mm256_unpackhi_epi16(a0, a4);
		b5 = simde_mm256_unpackhi_epi16(a1, a5);
		b6 = simde_mm256_unpackhi_epi16(a2, a6);
		b7 = simde_mm256_unpackhi_epi16(a3, a7);

		a0 = simde_mm256_unpacklo_epi16(b0, b2);
		a1 = simde_mm256_unpacklo_epi16(b1, b3);
		a2 = simde_mm256_unpackhi_epi16(b0, b2);
		a3 = simde_mm256_unpackhi_epi16(b1, b3);
		a4 = simde_mm256_unpacklo_epi16(b4, b6);
		a5 = simde_mm256_unpacklo_epi16(b5, b7);
		a6 = simde_mm256_unpackhi_epi16(b4, b6);
		a7 = simde_mm256_unpackhi_epi16(b5, b7);

		b0 = simde_mm256_unpacklo_epi16(a0, a1);
		b1 = simde_mm256_unpackhi_epi16(a0, a1);
		b2 = simde_mm256_unpacklo_epi16(a2, a3);
		b3 = simde_mm256_unpackhi_epi16(a2, a3);
		b4 = simde_mm256_unpacklo_epi16(a4, a5);
		b5 = simde_mm256_unpackhi_epi16(a4, a5);
		b6 = simde_mm256_unpacklo_epi16(a6, a7);
		b7 = simde_mm256_unpackhi_epi16(a6, a7);

		simde_mm_storeu_si128((simde__m128i *)(dst + 0 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 1 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 2 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 3 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 4 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 5 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 6 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 7 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 8 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b0, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 9 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b1, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 10 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b2, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 11 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b3, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 12 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b4, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 13 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b5, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 14 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b6, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 15 * dst_stride + 8 * i), simde_mm256_extracti128_si256(b7, 1));
	}
}
void PfreqN4SecTranspose32_AVX2_INTRIN(
	EB_S16 *src,
	EB_U32  src_stride,
	EB_S16 *dst,
	EB_U32  dst_stride)
{
	//EB_U32 i;
	// for (i = 0; i < 4; i++)
	//i = 0;
	{
		simde__m256i a0, a1, a2, a3, a4, a5, a6, a7;
		simde__m256i b0, b1, b2, b3, b4, b5, b6, b7;

		a0 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */0)*src_stride));
		a1 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */1)*src_stride));
		a2 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */2)*src_stride));
		a3 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */3)*src_stride));
		a4 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */4)*src_stride));
		a5 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */5)*src_stride));
		a6 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */6)*src_stride));
		a7 = simde_mm256_loadu_si256((const simde__m256i *)(src + (/*8 * i + */7)*src_stride));

		b0 = simde_mm256_unpacklo_epi16(a0, a4);
		b1 = simde_mm256_unpacklo_epi16(a1, a5);
		b2 = simde_mm256_unpacklo_epi16(a2, a6);
		b3 = simde_mm256_unpacklo_epi16(a3, a7);
		b4 = simde_mm256_unpackhi_epi16(a0, a4);
		b5 = simde_mm256_unpackhi_epi16(a1, a5);
		b6 = simde_mm256_unpackhi_epi16(a2, a6);
		b7 = simde_mm256_unpackhi_epi16(a3, a7);

		a0 = simde_mm256_unpacklo_epi16(b0, b2);
		a1 = simde_mm256_unpacklo_epi16(b1, b3);
		a2 = simde_mm256_unpackhi_epi16(b0, b2);
		a3 = simde_mm256_unpackhi_epi16(b1, b3);
		a4 = simde_mm256_unpacklo_epi16(b4, b6);
		a5 = simde_mm256_unpacklo_epi16(b5, b7);
		a6 = simde_mm256_unpackhi_epi16(b4, b6);
		a7 = simde_mm256_unpackhi_epi16(b5, b7);

		b0 = simde_mm256_unpacklo_epi16(a0, a1);
		b1 = simde_mm256_unpackhi_epi16(a0, a1);
		b2 = simde_mm256_unpacklo_epi16(a2, a3);
		b3 = simde_mm256_unpackhi_epi16(a2, a3);
		b4 = simde_mm256_unpacklo_epi16(a4, a5);
		b5 = simde_mm256_unpackhi_epi16(a4, a5);
		b6 = simde_mm256_unpacklo_epi16(a6, a7);
		b7 = simde_mm256_unpackhi_epi16(a6, a7);

		simde_mm_storeu_si128((simde__m128i *)(dst + 0 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b0, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 1 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b1, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 2 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b2, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 3 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b3, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 4 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b4, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 5 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b5, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 6 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b6, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 7 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b7, 0));
		simde_mm_storeu_si128((simde__m128i *)(dst + 8 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b0, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 9 * dst_stride /*+ 8 * i */), simde_mm256_extracti128_si256(b1, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 10 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b2, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 11 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b3, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 12 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b4, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 13 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b5, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 14 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b6, 1));
		simde_mm_storeu_si128((simde__m128i *)(dst + 15 * dst_stride/* + 8 * i*/), simde_mm256_extracti128_si256(b7, 1));
	}
}
void PfreqN4Transform32x32_AVX2_INTRIN(
	EB_S16 *src,
	const EB_U32 src_stride,
	EB_S16 *dst,
	const EB_U32 dst_stride,
	EB_S16 *intermediate,
	EB_U32 addshift)
{
	PfreqN41DTransform32_AVX2_INTRIN(src, src_stride, intermediate, 32, 6 + addshift);
	PfreqN4FirstTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);
	PfreqN42DTransform32_AVX2_INTRIN(dst, dst_stride, intermediate, 32, 9);
	PfreqN4SecTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);

	return;
}

// forward 16x16 transform
EB_EXTERN void lowPrecisionTransform16x16_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride, EB_S16 *intermediate, EB_U32 addshift)
{
    transform16_AVX2_INTRIN(src, src_stride, intermediate, 16, (EB_S16)(4 + addshift));
    EbHevcTranspose16_AVX2_INTRIN(intermediate, 16, dst, dst_stride);

    transform16_AVX2_INTRIN(dst, dst_stride, intermediate, 16, 9);
    EbHevcTranspose16_AVX2_INTRIN(intermediate, 16, dst, dst_stride);

}

// forward 32x32 transform
EB_EXTERN void lowPrecisionTransform32x32_AVX2_INTRIN(EB_S16 *src, EB_U32 src_stride, EB_S16 *dst, EB_U32 dst_stride, EB_S16 *intermediate, EB_U32 addshift)
{
    EbHevcTransform32_AVX2_INTRIN(src, src_stride, intermediate, 32, 6 + addshift);
    EbHevcTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);
    EbHevcTransform32_AVX2_INTRIN(dst, dst_stride, intermediate, 32, 9);
    EbHevcTranspose32_AVX2_INTRIN(intermediate, 32, dst, dst_stride);
}

void MatMult4x4_OutBuff_AVX2_INTRIN(
	EB_S16*              coeff,
	const EB_U32         coeffStride,
	EB_S16*              coeffOut,
	const EB_U32         coeffOutStride,
	const EB_U16        *maskingMatrix,
	const EB_U32         maskingMatrixStride,
	const EB_U32         computeSize,
	const EB_S32         offset,
	const EB_S32         shiftNum,
	EB_U32*              nonzerocoeff)

{
	simde__m256i z = simde_mm256_setzero_si256();
	simde__m128i a, b;
	simde__m256i coeffTemp, a0, a1, b0, b1, ymm_computed, MaskingMatrix, offsetREG, coeffTempORG;
	(void)computeSize;

	coeffTemp = a0 = a1 = b0 = b1 = ymm_computed = MaskingMatrix = offsetREG = simde_mm256_setzero_si256();

	// prepare Shift REG
	simde__m128i PMP_PRECISION_REG = simde_mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (EB_S16)shiftNum); //simde_mm_set1_epi16((EB_U16)shiftNum);//simde_mm_set1_epi32(shiftNum);

	//prepare the offset
	offsetREG = simde_mm256_set1_epi32(offset);

	//load maskingMatrix_new
	MaskingMatrix = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)maskingMatrix), simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + maskingMatrixStride)))), simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + 2 * maskingMatrixStride)), simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + 3 * maskingMatrixStride))), 0x1);

	//load coefftemp
	a = simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)coeff), simde_mm_loadl_epi64((simde__m128i*)(coeff + coeffStride))); // 1st and 2nd row of the 4x4 block
	b = simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)(coeff + 2 * coeffStride)), simde_mm_loadl_epi64((simde__m128i*)(coeff + 3 * coeffStride))); // 3rd and 4th row of the 4x4 block
	coeffTemp = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(a), b, 0x1); // the 4x4 block is now loaded

	coeffTempORG = coeffTemp;
	//Absolute val
	coeffTemp = simde_mm256_abs_epi16(coeffTemp);

	a0 = simde_mm256_mullo_epi16(coeffTemp, MaskingMatrix);
	a1 = simde_mm256_mulhi_epi16(coeffTemp, MaskingMatrix);


	b0 = simde_mm256_unpacklo_epi16(a0, a1);
	b1 = simde_mm256_unpackhi_epi16(a0, a1);

	b0 = simde_mm256_add_epi32(b0, offsetREG);
	b1 = simde_mm256_add_epi32(b1, offsetREG);

	//Shift right by PMP_PRECISION_REG
	b0 = simde_mm256_sra_epi32(b0, PMP_PRECISION_REG);
	b1 = simde_mm256_sra_epi32(b1, PMP_PRECISION_REG);

	//coefftemp in c
	ymm_computed = simde_mm256_packs_epi32(b0, b1);//Convert packed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
	z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(ymm_computed, simde_mm256_setzero_si256())); //coeffTemp = (coeff[coeffLocation] < 0)? -coeffTemp : coeffTemp;

	ymm_computed = simde_mm256_sign_epi16(ymm_computed, coeffTempORG);// coeffTemp);

	a = simde_mm256_extracti128_si256(ymm_computed, 0);
	b = simde_mm256_extracti128_si256(ymm_computed, 1);
	simde_mm_storel_epi64((simde__m128i *) coeffOut, a);
	simde_mm_storel_epi64((simde__m128i *)(coeffOut + 1 * coeffOutStride), simde_mm_srli_si128(a, 8));
	simde_mm_storel_epi64((simde__m128i *)(coeffOut + 2 * coeffOutStride), b);
	simde_mm_storel_epi64((simde__m128i *)(coeffOut + 3 * coeffOutStride), simde_mm_srli_si128(b, 8));

	z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 8));
	*nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));

}


/*****************************************************************************************************************************************************************/
void MatMult4x4_AVX2_INTRIN(
	EB_S16*              coeff,
	const EB_U32         coeffStride,
    const EB_U16        *maskingMatrix,
    const EB_U32         maskingMatrixStride,  //Matrix size
    const EB_U32         computeSize,  //Computation area size
	const EB_S32         offset,     //(PMP_MAX >> 1)
	const EB_S32         shiftNum, //PMP_PRECISION
	EB_U32*              nonzerocoeff)

{
		simde__m256i z = simde_mm256_setzero_si256();
		simde__m128i a, b;
		simde__m256i coeffTemp,a0,a1,b0,b1,ymm_computed,MaskingMatrix,offsetREG,coeffTempORG;
        (void)computeSize;

        coeffTemp = a0 = a1 = b0 = b1 = ymm_computed = MaskingMatrix = offsetREG = simde_mm256_setzero_si256();

		// prepare Shift REG
        simde__m128i PMP_PRECISION_REG = simde_mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (EB_S16)shiftNum); //simde_mm_set1_epi16((EB_U16)shiftNum);//simde_mm_set1_epi32(shiftNum);

		//prepare the offset
		offsetREG = simde_mm256_set1_epi32(offset);

		//load maskingMatrix_new
        MaskingMatrix = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)maskingMatrix), simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + maskingMatrixStride)))), simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + 2 * maskingMatrixStride)), simde_mm_loadl_epi64((simde__m128i*)(maskingMatrix + 3 * maskingMatrixStride))), 0x1);

		//load coefftemp
		a = simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)coeff), simde_mm_loadl_epi64((simde__m128i*)(coeff + coeffStride))); // 1st and 2nd row of the 4x4 block
		b = simde_mm_unpacklo_epi64(simde_mm_loadl_epi64((simde__m128i*)(coeff + 2 * coeffStride)), simde_mm_loadl_epi64((simde__m128i*)(coeff + 3 * coeffStride))); // 3rd and 4th row of the 4x4 block
		coeffTemp = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256( a),b,0x1); // the 4x4 block is now loaded

		coeffTempORG = coeffTemp;
        //Absolute val
		coeffTemp = simde_mm256_abs_epi16(coeffTemp);

		a0 = simde_mm256_mullo_epi16(coeffTemp, MaskingMatrix);
        a1 = simde_mm256_mulhi_epi16(coeffTemp, MaskingMatrix);


		b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);

		b0 = simde_mm256_add_epi32(b0, offsetREG);
        b1 = simde_mm256_add_epi32(b1, offsetREG);

		//Shift right by PMP_PRECISION_REG
		b0 = simde_mm256_sra_epi32(b0, PMP_PRECISION_REG);
        b1 = simde_mm256_sra_epi32(b1, PMP_PRECISION_REG);

		//coefftemp in c
		ymm_computed = simde_mm256_packs_epi32(b0, b1);//Convert packed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
		z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(ymm_computed, simde_mm256_setzero_si256())); //coeffTemp = (coeff[coeffLocation] < 0)? -coeffTemp : coeffTemp;

		ymm_computed = simde_mm256_sign_epi16(ymm_computed, coeffTempORG);// coeffTemp);

		a = simde_mm256_extracti128_si256(ymm_computed, 0);
        b = simde_mm256_extracti128_si256(ymm_computed, 1);
        simde_mm_storel_epi64((simde__m128i *)coeff, a);
        simde_mm_storel_epi64((simde__m128i *)(coeff + coeffStride), simde_mm_srli_si128(a, 8));
        simde_mm_storel_epi64((simde__m128i *)(coeff + 2 * coeffStride), b);
        simde_mm_storel_epi64((simde__m128i *)(coeff + 3 * coeffStride), simde_mm_srli_si128(b, 8));

		  z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 8));
          *nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));

}
/*******************************************MatMult8x8_AVX2_INTRIN**************************************************/
void MatMult8x8_AVX2_INTRIN(
	EB_S16*              coeff,
	const EB_U32         coeffStride,
    const EB_U16        *maskingMatrix,
    const EB_U32         maskingMatrixStride,  //Matrix size
    const EB_U32         computeSize,  //Computation area size
	const EB_S32         offset,     //(PMP_MAX >> 1)
	const EB_S32         shiftNum, //PMP_PRECISION
	EB_U32*              nonzerocoeff)
{
	unsigned row;
	simde__m256i z = simde_mm256_setzero_si256();
	//simde__m128i a, b;
	simde__m256i coeffTemp,a0,a1,b0,b1,ymm_computed,MaskingMatrix,coeffTempORG;

    coeffTemp = a0 = a1 = b0 = b1 = ymm_computed =MaskingMatrix=  simde_mm256_setzero_si256();

	// prepare Shift REG
    simde__m128i PMP_PRECISION_REG = simde_mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (EB_S16)shiftNum);//simde_mm_set1_epi32(shiftNum);

	//prepare the offset
	simde__m256i offsetREG = simde_mm256_set1_epi32(offset);
	row = 0;
    do {

		//load maskingMatrix_new
        MaskingMatrix = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((simde__m128i*)(maskingMatrix + maskingMatrixStride*row))), simde_mm_loadu_si128((simde__m128i*)(maskingMatrix + maskingMatrixStride*(row + 1))), 0x1);

		//load coefftemp
		coeffTemp = simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*row))),
                simde_mm_loadu_si128((simde__m128i*)(coeff + coeffStride*(row + 1))), 0x1);

		coeffTempORG = coeffTemp;
        //Absolute val
		coeffTemp = simde_mm256_abs_epi16(coeffTemp);

		//Multiply
		a0 = simde_mm256_mullo_epi16(coeffTemp, MaskingMatrix);
        a1 = simde_mm256_mulhi_epi16(coeffTemp, MaskingMatrix);


		b0 = simde_mm256_unpacklo_epi16(a0, a1);
        b1 = simde_mm256_unpackhi_epi16(a0, a1);

		//Add
		b0 = simde_mm256_add_epi32(b0, offsetREG);
        b1 = simde_mm256_add_epi32(b1, offsetREG);

		//Shift right by PMP_PRECISION_REG
        b0 = simde_mm256_sra_epi32(b0, PMP_PRECISION_REG);
        b1 = simde_mm256_sra_epi32(b1, PMP_PRECISION_REG);

		//coefftemp in c
		ymm_computed = simde_mm256_packs_epi32(b0, b1);//Convert packed 32-bit integers from b0 and b1 to packed 16-bit integers using signed saturation, and store the results in dst.
		z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(ymm_computed, simde_mm256_setzero_si256())); //coeffTemp = (coeff[coeffLocation] < 0)? -coeffTemp : coeffTemp;

		ymm_computed = simde_mm256_sign_epi16(ymm_computed, coeffTempORG);// coeffTemp);

		simde_mm_storeu_si128((simde__m128i *)(coeff + coeffStride*row), simde_mm256_extracti128_si256(ymm_computed, 0));
        simde_mm_storeu_si128((simde__m128i *)(coeff + coeffStride*(row + 1)), simde_mm256_extracti128_si256(ymm_computed, 1));

	    row += 2;
    } while (row < computeSize);

	z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 7));
    *nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));

}
/***************************************MatMultNxN_AVX2_INTRIN****************************************************/
void MatMultNxN_AVX2_INTRIN(
	EB_S16*              coeff,
	const EB_U32         coeffStride,
    const EB_U16        *maskingMatrix,
    const EB_U32         maskingMatrixStride,  //Matrix size
    const EB_U32         computeSize,  //Computation area size
	const EB_S32         offset,     //(PMP_MAX >> 1)
	const EB_S32         shiftNum, //PMP_PRECISION
	EB_U32*              nonzerocoeff)
{

	unsigned row,col;
	simde__m256i z = simde_mm256_setzero_si256();
	//simde__m128i a, b;
	simde__m256i coeffTemp,a0,a1,b0,b1,ymm_computed,MaskingMatrix,coeffTempORG;

    coeffTemp = a0 = a1 = b0 = b1 = ymm_computed =MaskingMatrix  = simde_mm256_setzero_si256();

	// prepare Shift REG
    simde__m128i PMP_PRECISION_REG = simde_mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (EB_S16)shiftNum);//simde_mm_set1_epi32(shiftNum);

	//prepare the offset
	simde__m256i offsetREG = simde_mm256_set1_epi32(offset);

	row = 0;
		do {
			col = 0;
			do {

				//load coefftemp
				coeffTemp = simde_mm256_loadu_si256((simde__m256i *)(coeff + coeffStride*row + col));

				//load maskingMatrix_new
                MaskingMatrix = simde_mm256_loadu_si256((simde__m256i *) (maskingMatrix + maskingMatrixStride*row + col));

                coeffTempORG = coeffTemp;

				//Absolute val
				coeffTemp = simde_mm256_abs_epi16(coeffTemp);

				//Multiply
				a0 = simde_mm256_mullo_epi16(coeffTemp, MaskingMatrix);
				a1 = simde_mm256_mulhi_epi16(coeffTemp, MaskingMatrix);


				b0 = simde_mm256_unpacklo_epi16(a0, a1);
				b1 = simde_mm256_unpackhi_epi16(a0, a1);

				//Add
				b0 = simde_mm256_add_epi32(b0, offsetREG);
				b1 = simde_mm256_add_epi32(b1, offsetREG);

				//Shift right by PMP_PRECISION_REG
				b0 = simde_mm256_sra_epi32(b0, PMP_PRECISION_REG);
				b1 = simde_mm256_sra_epi32(b1, PMP_PRECISION_REG);

				//coefftemp in c
				ymm_computed = simde_mm256_packs_epi32(b0, b1);//Convert packed 32-bit integers from b0 and b1 to packed 16-bit integers using signed saturation, and store the results in dst.
				z = simde_mm256_sub_epi16(z, simde_mm256_cmpgt_epi16(ymm_computed, simde_mm256_setzero_si256())); //coeffTemp = (coeff[coeffLocation] < 0)? -coeffTemp : coeffTemp;

				ymm_computed = simde_mm256_sign_epi16(ymm_computed, coeffTempORG);// coeffTemp);

				simde_mm256_storeu_si256((simde__m256i *)(coeff + coeffStride*row + col), ymm_computed);

			col += 16;
            } while (col < computeSize);

		row++;
        } while (row < computeSize);

	z = simde_mm256_sad_epu8(z, simde_mm256_srli_si256(z, 7));
    *nonzerocoeff = simde_mm_cvtsi128_si32(simde_mm_add_epi32(simde_mm256_extracti128_si256(z, 0), simde_mm256_extracti128_si256(z, 1)));

}
