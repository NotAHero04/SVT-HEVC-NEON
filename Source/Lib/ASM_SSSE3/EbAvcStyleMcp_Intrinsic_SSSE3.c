/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbAvcStyleMcp_SSSE3.h"

#include "EbDefinitions.h"

#include "../../../simde/simde/x86/ssse3.h"


EB_EXTERN EB_ALIGN(16) const EB_S8 EbHevcAvcStyleLumaIFCoeff8_SSSE3[]= {
    -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25,
     9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,
    -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18,
    18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2,
    -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9,
    25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1
};


void PictureCopyKernel_SSSE3(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight,
	EB_U32                   bytesPerSample);

void AvcStyleLumaInterpolationFilterHorizontal_SSSE3_INTRIN(
    EB_BYTE refPic,
    EB_U32 srcStride,
    EB_BYTE dst,
    EB_U32 dstStride,
    EB_U32 puWidth,
    EB_U32 puHeight,
    EB_BYTE tempBuf,
    EB_U32 fracPos)
{
    (void)tempBuf;
    simde__m128i IFOffset, IFCoeff_1_0, IFCoeff_3_2, sum_clip_U8;
    EB_U32 width_cnt, height_cnt;
    EB_U32 IFShift = 5;

    fracPos <<= 5;
    IFOffset = simde_mm_set1_epi16(0x0010);
    IFCoeff_1_0 = simde_mm_load_si128((simde__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 32));
    IFCoeff_3_2 = simde_mm_load_si128((simde__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 16));

    if (!(puWidth & 15)) { // 16x
        simde__m128i ref0, ref1, ref2, ref3, ref01_lo, ref01_hi, ref23_lo, ref23_hi, sum_lo, sum_hi;

        for (height_cnt = 0; height_cnt < puHeight; ++height_cnt){
            for (width_cnt = 0; width_cnt < puWidth; width_cnt += 16) {
                ref0 = simde_mm_loadu_si128((simde__m128i *)(refPic + width_cnt - 1));
                ref1 = simde_mm_loadu_si128((simde__m128i *)(refPic + width_cnt));
                ref2 = simde_mm_loadu_si128((simde__m128i *)(refPic + width_cnt + 1));
                ref3 = simde_mm_loadu_si128((simde__m128i *)(refPic + width_cnt + 2));

                ref01_lo = simde_mm_unpacklo_epi8(ref0, ref1);
                ref01_hi = simde_mm_unpackhi_epi8(ref0, ref1);
                ref23_lo = simde_mm_unpacklo_epi8(ref2, ref3);
                ref23_hi = simde_mm_unpackhi_epi8(ref2, ref3);

                sum_lo = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_maddubs_epi16(ref01_lo, IFCoeff_1_0), simde_mm_maddubs_epi16(ref23_lo, IFCoeff_3_2)), IFOffset), IFShift);
                sum_hi = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_maddubs_epi16(ref01_hi, IFCoeff_1_0), simde_mm_maddubs_epi16(ref23_hi, IFCoeff_3_2)), IFOffset), IFShift);
                sum_clip_U8 = simde_mm_packus_epi16(sum_lo, sum_hi);
                simde_mm_storeu_si128((simde__m128i *)(dst + width_cnt), sum_clip_U8);
            }
            refPic += srcStride;
            dst += dstStride;
        }
    }
    else { //8x
        simde__m128i  sum01, sum23, sum;

        for (height_cnt = 0; height_cnt < puHeight; ++height_cnt){
            for (width_cnt = 0; width_cnt < puWidth; width_cnt += 8) {
                sum01 = simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refPic + width_cnt - 1)),
                                                            simde_mm_loadl_epi64((simde__m128i *)(refPic + width_cnt))), IFCoeff_1_0);

                sum23 = simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refPic + width_cnt + 1)),
                                                            simde_mm_loadl_epi64((simde__m128i *)(refPic + width_cnt + 2))), IFCoeff_3_2);

                sum = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_add_epi16(sum01, sum23), IFOffset), IFShift);
                sum_clip_U8 = simde_mm_packus_epi16(sum, sum);

                simde_mm_storel_epi64((simde__m128i *)(dst + width_cnt), sum_clip_U8);
            }
            refPic += srcStride;
            dst += dstStride;
        }

    }
}

void AvcStyleLumaInterpolationFilterVertical_SSSE3_INTRIN(
    EB_BYTE refPic,
    EB_U32 srcStride,
    EB_BYTE dst,
    EB_U32 dstStride,
    EB_U32 puWidth,
    EB_U32 puHeight,
    EB_BYTE tempBuf,
    EB_U32 fracPos)
{
    (void)tempBuf;
    simde__m128i IFOffset, IFCoeff_1_0, IFCoeff_3_2, sum_clip_U8;
    EB_U32 width_cnt, height_cnt;
    EB_U32 IFShift = 5;
    EB_U32 srcStrideSkip = srcStride;
    EB_BYTE refPicTemp, dstTemp;

    fracPos <<= 5;
    refPic -= srcStride;
    IFOffset = simde_mm_set1_epi16(0x0010);
    IFCoeff_1_0 = simde_mm_load_si128((simde__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 32));
    IFCoeff_3_2 = simde_mm_load_si128((simde__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 16));
    if (!(puWidth & 15)) { //16x

        simde__m128i sum_lo, sum_hi, ref0, refs, ref2s, ref3s;

        for (width_cnt = 0; width_cnt < puWidth; width_cnt += 16) {

            refPicTemp = refPic;
            dstTemp = dst;

            for (height_cnt = 0; height_cnt < puHeight; ++height_cnt) {
                ref0 = simde_mm_loadu_si128((simde__m128i *)(refPicTemp));
                refs = simde_mm_loadu_si128((simde__m128i *)(refPicTemp + srcStride));
                ref2s = simde_mm_loadu_si128((simde__m128i *)(refPicTemp + 2 * srcStride));
                ref3s = simde_mm_loadu_si128((simde__m128i *)(refPicTemp + 3 * srcStride));

                sum_lo = simde_mm_add_epi16(simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(ref0, refs), IFCoeff_1_0),
                    simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(ref2s, ref3s), IFCoeff_3_2));

                sum_hi = simde_mm_add_epi16(simde_mm_maddubs_epi16(simde_mm_unpackhi_epi8(ref0, refs), IFCoeff_1_0),
                    simde_mm_maddubs_epi16(simde_mm_unpackhi_epi8(ref2s, ref3s), IFCoeff_3_2));

                sum_lo = simde_mm_srai_epi16(simde_mm_add_epi16(sum_lo, IFOffset), IFShift);
                sum_hi = simde_mm_srai_epi16(simde_mm_add_epi16(sum_hi, IFOffset), IFShift);
                sum_clip_U8 = simde_mm_packus_epi16(sum_lo, sum_hi);
                simde_mm_storeu_si128((simde__m128i *)(dstTemp), sum_clip_U8);
                dstTemp += dstStride;
                refPicTemp += srcStrideSkip;
            }
            refPic += 16;
            dst += 16;
        }
    }
    else { //8x
        simde__m128i sum, sum01, sum23;

        for (width_cnt = 0; width_cnt < puWidth; width_cnt += 8) {

            refPicTemp = refPic;
            dstTemp = dst;

            for (height_cnt = 0; height_cnt < puHeight; ++height_cnt) {
                sum01 = simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refPicTemp)),
                                                            simde_mm_loadl_epi64((simde__m128i *)(refPicTemp + srcStride))), IFCoeff_1_0);

                sum23 = simde_mm_maddubs_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refPicTemp + 2 * srcStride)),
                                                            simde_mm_loadl_epi64((simde__m128i *)(refPicTemp + 3 * srcStride))), IFCoeff_3_2);

                sum = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_add_epi16(sum01, sum23), IFOffset), IFShift);
                sum_clip_U8 = simde_mm_packus_epi16(sum, sum);
                simde_mm_storel_epi64((simde__m128i *)(dstTemp), sum_clip_U8);

                dstTemp += dstStride;
                refPicTemp += srcStrideSkip;
            }
            refPic += 8;
            dst += 8;
        }
    }
}
