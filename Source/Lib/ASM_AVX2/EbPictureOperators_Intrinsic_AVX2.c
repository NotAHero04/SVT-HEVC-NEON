/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/


#include "../../../simde/simde/x86/avx2.h"
#include "EbDefinitions.h"
#include "EbPictureOperators_AVX2.h"

#define simde_mm256_set_m128i(/* simde__m128i */ hi, /* simde__m128i */ lo) \
    simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(lo), (hi), 0x1)

void CompressedPackmsb_AVX2_INTRIN(
    EB_U8     *in8BitBuffer,
    EB_U32     in8Stride,
    EB_U8     *innBitBuffer,
    EB_U16    *out16BitBuffer,
    EB_U32     innStride,
    EB_U32     outStride,
    EB_U32     width,
    EB_U32     height)
{



    EB_U32 y;

    if (width == 8)
    {

        simde__m128i in2Bit, ext0, ext1, ext2, ext3, ext01, ext23, ext0_15, ext16_31;
        simde__m128i msk0, firstfour2Bit, secondfour2Bit, thirdfour2Bit, fourthfour2Bit;
        simde__m128i concat0_128, concat1_128, concat2_128, concat3_128, in8Bit_128_0, in8Bit_128_1, in8BitStride_128, in8Bit_128;
        msk0 = simde_mm_set1_epi8((signed char)0xC0);//1100.000


                                                //processing 2 lines for chroma
        for (y = 0; y < height; y += 4)
        {

            firstfour2Bit = simde_mm_loadl_epi64((simde__m128i*)innBitBuffer);
            secondfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + innStride));
            thirdfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + 2 * innStride));
            fourthfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + 3 * innStride));
            in2Bit = simde_mm_unpacklo_epi32(simde_mm_unpacklo_epi16(firstfour2Bit, secondfour2Bit), simde_mm_unpacklo_epi16(thirdfour2Bit, fourthfour2Bit));

            ext0 = simde_mm_and_si128(in2Bit, msk0);
            ext1 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 2), msk0);
            ext2 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 4), msk0);
            ext3 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 6), msk0);

            ext01 = simde_mm_unpacklo_epi8(ext0, ext1);
            ext23 = simde_mm_unpacklo_epi8(ext2, ext3);
            ext0_15 = simde_mm_unpacklo_epi16(ext01, ext23);
            ext16_31 = simde_mm_unpackhi_epi16(ext01, ext23);

            in8Bit_128_0 = simde_mm_loadl_epi64((simde__m128i*)in8BitBuffer);
            in8Bit_128_1 = simde_mm_loadl_epi64((simde__m128i*)(in8BitBuffer + in8Stride));
            in8Bit_128 = simde_mm_unpacklo_epi64(in8Bit_128_0, in8Bit_128_1);
            in8Bit_128_0 = simde_mm_loadl_epi64((simde__m128i*)(in8BitBuffer + 2 * in8Stride));
            in8Bit_128_1 = simde_mm_loadl_epi64((simde__m128i*)(in8BitBuffer + 3 * in8Stride));
            in8BitStride_128 = simde_mm_unpacklo_epi64(in8Bit_128_0, in8Bit_128_1);

            //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8
            concat0_128 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(ext0_15, in8Bit_128), 6);
            concat1_128 = simde_mm_srli_epi16(simde_mm_unpackhi_epi8(ext0_15, in8Bit_128), 6);
            concat2_128 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(ext16_31, in8BitStride_128), 6);
            concat3_128 = simde_mm_srli_epi16(simde_mm_unpackhi_epi8(ext16_31, in8BitStride_128), 6);

            simde_mm_store_si128((simde__m128i*) out16BitBuffer, concat0_128);
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride), concat1_128);
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 2 * outStride), concat2_128);
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 3 * outStride), concat3_128);


            in8BitBuffer += in8Stride << 2;
            innBitBuffer += innStride << 2;
            out16BitBuffer += outStride << 2;
        }
    }
    if (width == 16)
    {
        simde__m256i inNBit, inNBitStride, in8Bit, in8BitStride, concat0, concat1, concat2, concat3;
        simde__m128i in2Bit, ext0, ext1, ext2, ext3, ext01, ext23, ext01h, ext23h, ext0_15, ext16_31, ext32_47, ext48_63;
        simde__m128i msk0, firstfour2Bit, secondfour2Bit, thirdfour2Bit, fourthfour2Bit;
        simde__m128i  in8Bit_128_0, in8Bit_128_1;
        msk0 = simde_mm_set1_epi8((signed char)0xC0);//1100.000

                                                //processing 2 lines for chroma
        for (y = 0; y < height; y += 4)
        {

            firstfour2Bit = simde_mm_loadl_epi64((simde__m128i*)innBitBuffer);
            secondfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + innStride));
            thirdfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + 2 * innStride));
            fourthfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + 3 * innStride));
            in2Bit = simde_mm_unpacklo_epi64(simde_mm_unpacklo_epi32(firstfour2Bit, secondfour2Bit), simde_mm_unpacklo_epi32(thirdfour2Bit, fourthfour2Bit));

            ext0 = simde_mm_and_si128(in2Bit, msk0);
            ext1 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 2), msk0);
            ext2 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 4), msk0);
            ext3 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 6), msk0);

            ext01 = simde_mm_unpacklo_epi8(ext0, ext1);
            ext23 = simde_mm_unpacklo_epi8(ext2, ext3);
            ext0_15 = simde_mm_unpacklo_epi16(ext01, ext23);
            ext16_31 = simde_mm_unpackhi_epi16(ext01, ext23);

            ext01h = simde_mm_unpackhi_epi8(ext0, ext1);
            ext23h = simde_mm_unpackhi_epi8(ext2, ext3);
            ext32_47 = simde_mm_unpacklo_epi16(ext01h, ext23h);
            ext48_63 = simde_mm_unpackhi_epi16(ext01h, ext23h);

            inNBit = simde_mm256_set_m128i(ext16_31, ext0_15);
            inNBitStride = simde_mm256_set_m128i(ext48_63, ext32_47);

            in8Bit_128_0 = simde_mm_loadu_si128((simde__m128i*)in8BitBuffer);
            in8Bit_128_1 = simde_mm_loadu_si128((simde__m128i*)(in8BitBuffer + in8Stride));
            in8Bit = simde_mm256_set_m128i(in8Bit_128_1, in8Bit_128_0);
            in8Bit_128_0 = simde_mm_loadu_si128((simde__m128i*)(in8BitBuffer + 2 * in8Stride));
            in8Bit_128_1 = simde_mm_loadu_si128((simde__m128i*)(in8BitBuffer + 3 * in8Stride));
            in8BitStride = simde_mm256_set_m128i(in8Bit_128_1, in8Bit_128_0);


            //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8
            concat0 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit, in8Bit), 6);
            concat1 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit, in8Bit), 6);
            concat2 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBitStride, in8BitStride), 6);
            concat3 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBitStride, in8BitStride), 6);
            simde_mm_store_si128((simde__m128i*) out16BitBuffer, simde_mm256_castsi256_si128(concat0));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 8), simde_mm256_castsi256_si128(concat1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride), simde_mm256_extracti128_si256(concat0, 1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride + 8), simde_mm256_extracti128_si256(concat1, 1));

            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 2 * outStride), simde_mm256_castsi256_si128(concat2));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 2 * outStride + 8), simde_mm256_castsi256_si128(concat3));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 3 * outStride), simde_mm256_extracti128_si256(concat2, 1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 3 * outStride + 8), simde_mm256_extracti128_si256(concat3, 1));
            in8BitBuffer += in8Stride << 2;
            innBitBuffer += innStride << 2;
            out16BitBuffer += outStride << 2;
        }
    }
    if (width == 32)
    {
        simde__m256i inNBit, in8Bit, inNBitStride, in8BitStride, concat0, concat1, concat2, concat3;
        simde__m128i in2Bit, ext0, ext1, ext2, ext3, ext01, ext23, ext01h, ext23h, ext0_15, ext16_31, ext32_47, ext48_63;
        simde__m128i msk0, firstfour2Bit, secondfour2Bit;

        msk0 = simde_mm_set1_epi8((signed char)0xC0);//1100.000

                                                //processing 2 lines for chroma
        for (y = 0; y < height; y += 2)
        {
            firstfour2Bit = simde_mm_loadl_epi64((simde__m128i*)innBitBuffer);
            secondfour2Bit = simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + innStride));
            in2Bit = simde_mm_unpacklo_epi64(firstfour2Bit, secondfour2Bit);

            ext0 = simde_mm_and_si128(in2Bit, msk0);
            ext1 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 2), msk0);
            ext2 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 4), msk0);
            ext3 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 6), msk0);

            ext01 = simde_mm_unpacklo_epi8(ext0, ext1);
            ext23 = simde_mm_unpacklo_epi8(ext2, ext3);
            ext0_15 = simde_mm_unpacklo_epi16(ext01, ext23);
            ext16_31 = simde_mm_unpackhi_epi16(ext01, ext23);

            ext01h = simde_mm_unpackhi_epi8(ext0, ext1);
            ext23h = simde_mm_unpackhi_epi8(ext2, ext3);
            ext32_47 = simde_mm_unpacklo_epi16(ext01h, ext23h);
            ext48_63 = simde_mm_unpackhi_epi16(ext01h, ext23h);

            inNBit = simde_mm256_set_m128i(ext16_31, ext0_15);
            inNBitStride = simde_mm256_set_m128i(ext48_63, ext32_47);

            in8Bit = simde_mm256_loadu_si256((simde__m256i*)in8BitBuffer);
            in8BitStride = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + in8Stride));


            //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8
            concat0 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit, in8Bit), 6);
            concat1 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit, in8Bit), 6);
            concat2 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBitStride, in8BitStride), 6);
            concat3 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBitStride, in8BitStride), 6);
            simde_mm_store_si128((simde__m128i*) out16BitBuffer, simde_mm256_castsi256_si128(concat0));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 8), simde_mm256_castsi256_si128(concat1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 16), simde_mm256_extracti128_si256(concat0, 1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + 24), simde_mm256_extracti128_si256(concat1, 1));

            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride), simde_mm256_castsi256_si128(concat2));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride + 8), simde_mm256_castsi256_si128(concat3));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride + 16), simde_mm256_extracti128_si256(concat2, 1));
            simde_mm_store_si128((simde__m128i*) (out16BitBuffer + outStride + 24), simde_mm256_extracti128_si256(concat3, 1));

            in8BitBuffer += in8Stride << 1;
            innBitBuffer += innStride << 1;
            out16BitBuffer += outStride << 1;
        }
    }
    else if (width == 64)
    {
        simde__m256i inNBit, in8Bit, inNBit32, in8Bit32;
        simde__m256i concat0, concat1, concat2, concat3;
        simde__m128i in2Bit, ext0, ext1, ext2, ext3, ext01, ext23, ext01h, ext23h, ext0_15, ext16_31, ext32_47, ext48_63;
        simde__m128i msk;

        msk = simde_mm_set1_epi8((signed char)0xC0);//1100.000

                                               //One row per iter
        for (y = 0; y < height; y++)
        {

            in2Bit = simde_mm_loadu_si128((simde__m128i*)innBitBuffer);

            ext0 = simde_mm_and_si128(in2Bit, msk);
            ext1 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 2), msk);
            ext2 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 4), msk);
            ext3 = simde_mm_and_si128(simde_mm_slli_epi16(in2Bit, 6), msk);

            ext01 = simde_mm_unpacklo_epi8(ext0, ext1);
            ext23 = simde_mm_unpacklo_epi8(ext2, ext3);
            ext0_15 = simde_mm_unpacklo_epi16(ext01, ext23);
            ext16_31 = simde_mm_unpackhi_epi16(ext01, ext23);

            ext01h = simde_mm_unpackhi_epi8(ext0, ext1);
            ext23h = simde_mm_unpackhi_epi8(ext2, ext3);
            ext32_47 = simde_mm_unpacklo_epi16(ext01h, ext23h);
            ext48_63 = simde_mm_unpackhi_epi16(ext01h, ext23h);

            inNBit = simde_mm256_set_m128i(ext16_31, ext0_15);
            inNBit32 = simde_mm256_set_m128i(ext48_63, ext32_47);

            in8Bit = simde_mm256_loadu_si256((simde__m256i*)in8BitBuffer);
            in8Bit32 = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + 32));

            //(outPixel | nBitPixel) concatenation 
            concat0 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit, in8Bit), 6);
            concat1 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit, in8Bit), 6);
            concat2 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit32, in8Bit32), 6);
            concat3 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit32, in8Bit32), 6);
            
            simde_mm_storeu_si128((simde__m128i*) out16BitBuffer, simde_mm256_castsi256_si128(concat0));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 8), simde_mm256_castsi256_si128(concat1));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 16), simde_mm256_extracti128_si256(concat0, 1));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 24), simde_mm256_extracti128_si256(concat1, 1));

            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 32), simde_mm256_castsi256_si128(concat2));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 40), simde_mm256_castsi256_si128(concat3));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 48), simde_mm256_extracti128_si256(concat2, 1));
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 56), simde_mm256_extracti128_si256(concat3, 1));


            in8BitBuffer += in8Stride;
            innBitBuffer += innStride;
            out16BitBuffer += outStride;

        }

    }

}



void CPack_AVX2_INTRIN(
	const EB_U8     *innBitBuffer,
	EB_U32     innStride,
	EB_U8     *inCompnBitBuffer,
	EB_U32     outStride,
	EB_U8    *localCache,

	EB_U32     width,
	EB_U32     height)
{

	EB_U32 y;

	if (width == 32)
	{
		simde__m256i inNBit;

		simde__m256i ext0, ext1, ext2, ext3, ext0123, ext0123n, extp;
		simde__m256i msk0, msk1, msk2, msk3;

		msk0 = simde_mm256_set1_epi32(0x000000C0);//1100.0000
		msk1 = simde_mm256_set1_epi32(0x00000030);//0011.0000
		msk2 = simde_mm256_set1_epi32(0x0000000C);//0000.1100
		msk3 = simde_mm256_set1_epi32(0x00000003);//0000.0011

		//One row per iter
		for (y = 0; y < height; y++)
		{


			inNBit = simde_mm256_loadu_si256((simde__m256i*)innBitBuffer);

			ext0 = simde_mm256_and_si256(inNBit, msk0);
			ext1 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 1 * 8 + 2), msk1);
			ext2 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 2 * 8 + 4), msk2);
			ext3 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 3 * 8 + 6), msk3);

			ext0123 = simde_mm256_or_si256(simde_mm256_or_si256(ext0, ext1), simde_mm256_or_si256(ext2, ext3));

			ext0123n = simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(ext0123, 1));


			extp = simde_mm256_packus_epi32(ext0123, ext0123n);
			extp = simde_mm256_packus_epi16(extp, extp);

			simde_mm_storel_epi64((simde__m128i*) inCompnBitBuffer, simde_mm256_castsi256_si128(extp));
			inCompnBitBuffer += 8;
			innBitBuffer += innStride;


		}

	}
	else if (width == 64)
	{
		simde__m256i inNBit;
		simde__m256i ext0, ext1, ext2, ext3, ext0123, ext0123n, extp, extp1;
		simde__m256i msk0, msk1, msk2, msk3;

		msk0 = simde_mm256_set1_epi32(0x000000C0);//1100.0000
		msk1 = simde_mm256_set1_epi32(0x00000030);//0011.0000
		msk2 = simde_mm256_set1_epi32(0x0000000C);//0000.1100
		msk3 = simde_mm256_set1_epi32(0x00000003);//0000.0011
		if (height == 64)
		{

			EB_U8* localPtr = localCache;


			for (y = 0; y < height; y++)
			{


				inNBit = simde_mm256_loadu_si256((simde__m256i*)innBitBuffer);


				ext0 = simde_mm256_and_si256(inNBit, msk0);
				ext1 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 1 * 8 + 2), msk1);
				ext2 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 2 * 8 + 4), msk2);
				ext3 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 3 * 8 + 6), msk3);

				ext0123 = simde_mm256_or_si256(simde_mm256_or_si256(ext0, ext1), simde_mm256_or_si256(ext2, ext3));

				ext0123n = simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(ext0123, 1));

				extp = simde_mm256_packus_epi32(ext0123, ext0123n);
				extp = simde_mm256_packus_epi16(extp, extp);


				inNBit = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + 32));

				ext0 = simde_mm256_and_si256(inNBit, msk0);
				ext1 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 1 * 8 + 2), msk1);
				ext2 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 2 * 8 + 4), msk2);
				ext3 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 3 * 8 + 6), msk3);

				ext0123 = simde_mm256_or_si256(simde_mm256_or_si256(ext0, ext1), simde_mm256_or_si256(ext2, ext3));

				ext0123n = simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(ext0123, 1));


				extp1 = simde_mm256_packus_epi32(ext0123, ext0123n);
				extp1 = simde_mm256_packus_epi16(extp1, extp1);

				extp = simde_mm256_unpacklo_epi64(extp, extp1);

				simde_mm_storeu_si128((simde__m128i*)  (localPtr + 16 * (y & 3)), simde_mm256_castsi256_si128(extp));

				if ((y & 3) == 3)
				{
					simde__m256i c0 = simde_mm256_loadu_si256((simde__m256i*)(localPtr));
					simde__m256i c1 = simde_mm256_loadu_si256((simde__m256i*)(localPtr + 32));
					simde_mm256_stream_si256((simde__m256i*)&inCompnBitBuffer[0], c0);
					simde_mm256_stream_si256((simde__m256i*)&inCompnBitBuffer[32], c1);
					inCompnBitBuffer += 4 * outStride;
				}

				innBitBuffer += innStride;

			}

		}
		else{

			//One row per iter
			for (y = 0; y < height; y++)
			{


				inNBit = simde_mm256_loadu_si256((simde__m256i*)innBitBuffer);


				ext0 = simde_mm256_and_si256(inNBit, msk0);
				ext1 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 1 * 8 + 2), msk1);
				ext2 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 2 * 8 + 4), msk2);
				ext3 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 3 * 8 + 6), msk3);

				ext0123 = simde_mm256_or_si256(simde_mm256_or_si256(ext0, ext1), simde_mm256_or_si256(ext2, ext3));

				ext0123n = simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(ext0123, 1));

				extp = simde_mm256_packus_epi32(ext0123, ext0123n);
				extp = simde_mm256_packus_epi16(extp, extp);


				inNBit = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + 32));

				ext0 = simde_mm256_and_si256(inNBit, msk0);
				ext1 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 1 * 8 + 2), msk1);
				ext2 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 2 * 8 + 4), msk2);
				ext3 = simde_mm256_and_si256(simde_mm256_srli_epi32(inNBit, 3 * 8 + 6), msk3);

				ext0123 = simde_mm256_or_si256(simde_mm256_or_si256(ext0, ext1), simde_mm256_or_si256(ext2, ext3));

				ext0123n = simde_mm256_castsi128_si256(simde_mm256_extracti128_si256(ext0123, 1));


				extp1 = simde_mm256_packus_epi32(ext0123, ext0123n);
				extp1 = simde_mm256_packus_epi16(extp1, extp1);

				extp = simde_mm256_unpacklo_epi64(extp, extp1);

				simde_mm_storeu_si128((simde__m128i*)  inCompnBitBuffer, simde_mm256_castsi256_si128(extp));

				inCompnBitBuffer += outStride;

				innBitBuffer += innStride;

			}

		}

	}

}


void EB_ENC_msbPack2D_AVX2_INTRIN_AL(
	EB_U8     *in8BitBuffer,
	EB_U32     in8Stride,
	EB_U8     *innBitBuffer,
	EB_U16    *out16BitBuffer,
	EB_U32     innStride,
	EB_U32     outStride,
	EB_U32     width,
	EB_U32     height)
{
    //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8

	EB_U32 y, x;

    simde__m128i out0, out1;


	if (width == 4)
	{
		for (y = 0; y < height; y += 2){

            out0 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)innBitBuffer), simde_mm_cvtsi32_si128(*(EB_U32 *)in8BitBuffer)), 6);
            out1 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)(innBitBuffer + innStride)), simde_mm_cvtsi32_si128(*(EB_U32 *)(in8BitBuffer + in8Stride))), 6);

            simde_mm_storel_epi64((simde__m128i*) out16BitBuffer, out0);
            simde_mm_storel_epi64((simde__m128i*) (out16BitBuffer + outStride), out1);

			in8BitBuffer += in8Stride << 1;
			innBitBuffer += innStride << 1;
			out16BitBuffer += outStride << 1;
		}
	}
	else if (width == 8)
	{
		for (y = 0; y < height; y += 2){

			out0 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i*)innBitBuffer), simde_mm_loadl_epi64((simde__m128i*)in8BitBuffer)), 6);
			out1 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + innStride)), simde_mm_loadl_epi64((simde__m128i*)(in8BitBuffer + in8Stride))), 6);

            simde_mm_storeu_si128((simde__m128i*) out16BitBuffer, out0);
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + outStride), out1);

			in8BitBuffer += in8Stride << 1;
			innBitBuffer += innStride << 1;
			out16BitBuffer += outStride << 1;
		}
	}
	else if (width == 16)
	{
        simde__m128i inNBit, in8Bit, inNBitStride, in8BitStride, out0, out1, out2, out3;

		for (y = 0; y < height; y += 2){
            inNBit = simde_mm_loadu_si128((simde__m128i*)innBitBuffer);
            in8Bit = simde_mm_loadu_si128((simde__m128i*)in8BitBuffer);
            inNBitStride = simde_mm_loadu_si128((simde__m128i*)(innBitBuffer + innStride));
            in8BitStride = simde_mm_loadu_si128((simde__m128i*)(in8BitBuffer + in8Stride));

            out0 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(inNBit, in8Bit), 6);
            out1 = simde_mm_srli_epi16(simde_mm_unpackhi_epi8(inNBit, in8Bit), 6);
            out2 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(inNBitStride, in8BitStride), 6);
            out3 = simde_mm_srli_epi16(simde_mm_unpackhi_epi8(inNBitStride, in8BitStride), 6);

            simde_mm_storeu_si128((simde__m128i*) out16BitBuffer, out0);
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + 8), out1);
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + outStride), out2);
            simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + outStride + 8), out3);

			in8BitBuffer += in8Stride << 1;
			innBitBuffer += innStride << 1;
			out16BitBuffer += outStride << 1;
		}
	}
	else if (width == 32)
	{
        simde__m256i inNBit, in8Bit, inNBitStride, in8BitStride, concat0, concat1, concat2, concat3;
        simde__m256i out0_15, out16_31, out_s0_s15, out_s16_s31;

		for (y = 0; y < height; y += 2){

            inNBit = simde_mm256_loadu_si256((simde__m256i*)innBitBuffer);
            in8Bit = simde_mm256_loadu_si256((simde__m256i*)in8BitBuffer);
            inNBitStride = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + innStride));
            in8BitStride = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + in8Stride));

            //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8
            concat0 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit, in8Bit), 6);
            concat1 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit, in8Bit), 6);
            concat2 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBitStride, in8BitStride), 6);
            concat3 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBitStride, in8BitStride), 6);

            //Re-organize the packing for writing to the out buffer
            out0_15 = simde_mm256_inserti128_si256(concat0, simde_mm256_extracti128_si256(concat1, 0), 1);
            out16_31 = simde_mm256_inserti128_si256(concat1, simde_mm256_extracti128_si256(concat0, 1), 0);
            out_s0_s15 = simde_mm256_inserti128_si256(concat2, simde_mm256_extracti128_si256(concat3, 0), 1);
            out_s16_s31 = simde_mm256_inserti128_si256(concat3, simde_mm256_extracti128_si256(concat2, 1), 0);

            simde_mm256_store_si256((simde__m256i*) out16BitBuffer, out0_15);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + 16), out16_31);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride), out_s0_s15);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride + 16), out_s16_s31);

			in8BitBuffer += in8Stride << 1;
			//innBitBuffer += innStride << 1;
            innBitBuffer += innStride *2;
			out16BitBuffer += outStride << 1;
		}
	}
	else if (width == 64)
	{
        simde__m256i inNBit, in8Bit, inNBitStride, in8BitStride, inNBit32, in8Bit32, inNBitStride32, in8BitStride32;
        simde__m256i concat0, concat1, concat2, concat3, concat4, concat5, concat6, concat7;
        simde__m256i out_0_15, out16_31, out32_47, out_48_63, out_s0_s15, out_s16_s31, out_s32_s47, out_s48_s63;

		for (y = 0; y < height; y += 2){

            inNBit = simde_mm256_loadu_si256((simde__m256i*)innBitBuffer);
            in8Bit = simde_mm256_loadu_si256((simde__m256i*)in8BitBuffer);
            inNBit32 = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + 32));
            in8Bit32 = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + 32));
            inNBitStride = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + innStride));
            in8BitStride = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + in8Stride));
            inNBitStride32 = simde_mm256_loadu_si256((simde__m256i*)(innBitBuffer + innStride + 32));
            in8BitStride32 = simde_mm256_loadu_si256((simde__m256i*)(in8BitBuffer + in8Stride + 32));
            //(outPixel | nBitPixel) concatenation is done with unpacklo_epi8 and unpackhi_epi8
            concat0 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit, in8Bit), 6);
            concat1 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit, in8Bit), 6);
            concat2 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBit32, in8Bit32), 6);
            concat3 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBit32, in8Bit32), 6);
            concat4 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBitStride, in8BitStride), 6);
            concat5 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBitStride, in8BitStride), 6);
            concat6 = simde_mm256_srli_epi16(simde_mm256_unpacklo_epi8(inNBitStride32, in8BitStride32), 6);
            concat7 = simde_mm256_srli_epi16(simde_mm256_unpackhi_epi8(inNBitStride32, in8BitStride32), 6);

            //Re-organize the packing for writing to the out buffer
            out_0_15 = simde_mm256_inserti128_si256(concat0, simde_mm256_extracti128_si256(concat1, 0), 1);
            out16_31 = simde_mm256_inserti128_si256(concat1, simde_mm256_extracti128_si256(concat0, 1), 0);
            out32_47 = simde_mm256_inserti128_si256(concat2, simde_mm256_extracti128_si256(concat3, 0), 1);
            out_48_63 = simde_mm256_inserti128_si256(concat3, simde_mm256_extracti128_si256(concat2, 1), 0);
            out_s0_s15 = simde_mm256_inserti128_si256(concat4, simde_mm256_extracti128_si256(concat5, 0), 1);
            out_s16_s31 = simde_mm256_inserti128_si256(concat5, simde_mm256_extracti128_si256(concat4, 1), 0);
            out_s32_s47 = simde_mm256_inserti128_si256(concat6, simde_mm256_extracti128_si256(concat7, 0), 1);
            out_s48_s63 = simde_mm256_inserti128_si256(concat7, simde_mm256_extracti128_si256(concat6, 1), 0);

            simde_mm256_store_si256((simde__m256i*) out16BitBuffer, out_0_15);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + 16), out16_31);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + 32), out32_47);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + 48), out_48_63);

            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride), out_s0_s15);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride + 16), out_s16_s31);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride + 32), out_s32_s47);
            simde_mm256_store_si256((simde__m256i*) (out16BitBuffer + outStride + 48), out_s48_s63);

			in8BitBuffer += in8Stride << 1;
			//innBitBuffer += innStride << 1;
            innBitBuffer += innStride *2;
			out16BitBuffer += outStride << 1;
		}
	}
	else
	{
        EB_U32 innStrideDiff = 2 * innStride;
		EB_U32 in8StrideDiff = 2 * in8Stride;
		EB_U32 outStrideDiff = 2 * outStride;
		innStrideDiff -= width;
		in8StrideDiff -= width;
		outStrideDiff -= width;

		if (!(width & 7)){

			for (x = 0; x < height; x += 2){
				for (y = 0; y < width; y += 8){

					out0 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i*)innBitBuffer), simde_mm_loadl_epi64((simde__m128i*)in8BitBuffer)), 6);
					out1 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i*)(innBitBuffer + innStride)), simde_mm_loadl_epi64((simde__m128i*)(in8BitBuffer + in8Stride))), 6);

					simde_mm_storeu_si128((simde__m128i*) out16BitBuffer, out0);
					simde_mm_storeu_si128((simde__m128i*) (out16BitBuffer + outStride), out1);

					in8BitBuffer += 8;
					innBitBuffer += 8;
					out16BitBuffer += 8;
				}
				in8BitBuffer += in8StrideDiff;
				innBitBuffer += innStrideDiff;
				out16BitBuffer += outStrideDiff;
			}
		}
		else{
			for (x = 0; x < height; x += 2){
				for (y = 0; y < width; y += 4){

					out0 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)innBitBuffer), simde_mm_cvtsi32_si128(*(EB_U32 *)in8BitBuffer)), 6);
					out1 = simde_mm_srli_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)(innBitBuffer + innStride)), simde_mm_cvtsi32_si128(*(EB_U32 *)(in8BitBuffer + in8Stride))), 6);

					simde_mm_storel_epi64((simde__m128i*) out16BitBuffer, out0);
					simde_mm_storel_epi64((simde__m128i*) (out16BitBuffer + outStride), out1);

					in8BitBuffer += 4;
					innBitBuffer += 4;
					out16BitBuffer += 4;
				}
				in8BitBuffer += in8StrideDiff;
				innBitBuffer += innStrideDiff;
				out16BitBuffer += outStrideDiff;
			}
		}
	}
}

#define ALSTORE  1
#define B256     1

void UnpackAvg_AVX2_INTRIN(
	EB_U16 *ref16L0,
        EB_U32  refL0Stride,
        EB_U16 *ref16L1,
        EB_U32  refL1Stride,
        EB_U8  *dstPtr,
        EB_U32  dstStride,      
        EB_U32  width,
        EB_U32  height)
{

	EB_U32   y;
    simde__m128i inPixel0, inPixel1;   	

  

	if (width == 4)
	{
         simde__m128i out8_0_U8_L0, out8_0_U8_L1;
        simde__m128i avg8_0_U8;
		
        for (y = 0; y < height; y += 2)
		{
            //--------
            //Line One
            //--------

            //List0
            inPixel0 = simde_mm_loadl_epi64((simde__m128i*)ref16L0);
            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //List1
            inPixel0 = simde_mm_loadl_epi64((simde__m128i*)ref16L1);
            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );	

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           

            *(EB_U32*)dstPtr = simde_mm_cvtsi128_si32(avg8_0_U8);

			//--------
            //Line Two
            //--------

            //List0
            inPixel0 = simde_mm_loadl_epi64((simde__m128i*)(ref16L0+ refL0Stride) );
            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //List1

            inPixel0 = simde_mm_loadl_epi64((simde__m128i*)(ref16L1+ refL1Stride) );
            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );	

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           

            *(EB_U32*)(dstPtr+dstStride) = simde_mm_cvtsi128_si32(avg8_0_U8);

            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;

		}
		
	}
	else if (width == 8)
	{
		
        simde__m128i out8_0_U8_L0, out8_0_U8_L1, out8_2_U8_L0,out8_2_U8_L1;
        simde__m128i avg8_0_U8,avg8_2_U8;
        
        for (y = 0; y < height; y += 2)
		{
            //--------
            //Line One
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L0); 

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

             //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;
            out8_0_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           

			simde_mm_storel_epi64((simde__m128i*) dstPtr      , avg8_0_U8);
		

            //--------
            //Line Two
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + refL0Stride) );

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_2_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + refL1Stride) );

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;
            out8_2_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //AVG
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);           

			simde_mm_storel_epi64((simde__m128i*)(dstPtr +dstStride)    , avg8_2_U8);

            	
            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;
        }
		
	}
	else if (width == 16)
	{

        simde__m128i inPixel4, inPixel5;
        simde__m128i out8_0_U8_L0, out8_0_U8_L1, out8_2_U8_L0,out8_2_U8_L1;
        simde__m128i avg8_0_U8,avg8_2_U8;

        for (y = 0; y < height; y += 2)
		{
            //--------
            //Line One
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)  ref16L0);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + 8));
      
            out8_0_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );		

             //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 8));          
    
            out8_0_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
	

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           
#if ALSTORE
            simde_mm_store_si128((simde__m128i*) dstPtr      , avg8_0_U8);
#else
			simde_mm_storeu_si128((simde__m128i*) dstPtr      , avg8_0_U8);
#endif	

            //--------
            //Line Two
            //--------

            //List0

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride + 8));         

           	out8_2_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
          
            //List1

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride + 8));          

           	out8_2_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
         
            
            //AVG
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);               
#if ALSTORE
            simde_mm_store_si128((simde__m128i*)(dstPtr  + dstStride      ) , avg8_2_U8);	
#else
			simde_mm_storeu_si128((simde__m128i*)(dstPtr  + dstStride      ) , avg8_2_U8);		
#endif
            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;

        }
       
	}
	else if (width == 32)
	{

#if B256
        simde__m256i inVal16b_0,inVal16b_1;
        simde__m256i data8b_32_0_L0,data8b_32_0_L1;
        simde__m256i avg8b_32_0;
#else
        simde__m128i inPixel2, inPixel3, inPixel4, inPixel5, inPixel6, inPixel7;
        simde__m128i out8_0_U8_L0, out8_1_U8_L0, out8_2_U8_L0, out8_3_U8_L0;
        simde__m128i out8_0_U8_L1, out8_1_U8_L1, out8_2_U8_L1, out8_3_U8_L1;
        simde__m128i avg8_0_U8, avg8_1_U8, avg8_2_U8, avg8_3_U8;
#endif       

       for (y = 0; y < height; y += 2)
		{

#if B256
            //--------
            //Line One
            //--------

            //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L0);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 16));         
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));            
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L1);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 16));          
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
                             
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);          

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);
                     
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr     ), avg8b_32_0);
           
            //--------
            //Line Two
            //--------
              //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + refL0Stride     ));
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + refL0Stride + 16));
         
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
         
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + refL1Stride     ));
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + refL1Stride + 16));
           
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
                                
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);           

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);          
           
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr + dstStride   ), avg8b_32_0);
           
#else
            //--------
            //Line One
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)  ref16L0);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + 8));
            inPixel2 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + 16));
            inPixel3 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + 24));

            out8_0_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
			out8_1_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel2, 2)  ,  simde_mm_srli_epi16(inPixel3, 2)  );

             //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 8));
            inPixel2 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 16));
            inPixel3 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 24));
     
            out8_0_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
			out8_1_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel2, 2)  ,  simde_mm_srli_epi16(inPixel3, 2)  );

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);
            avg8_1_U8 =  simde_mm_avg_epu8 (out8_1_U8_L0 , out8_1_U8_L1);         
#if ALSTORE
            simde_mm_store_si128((simde__m128i*) dstPtr      , avg8_0_U8);
			simde_mm_store_si128((simde__m128i*)(dstPtr + 16), avg8_1_U8);
#else
			simde_mm_storeu_si128((simde__m128i*) dstPtr      , avg8_0_U8);
			simde_mm_storeu_si128((simde__m128i*)(dstPtr + 16), avg8_1_U8);
#endif

            //--------
            //Line Two
            //--------

            //List0

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride + 8));
            inPixel6 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride + 16));
            inPixel7 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride + 24));

           	out8_2_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
            out8_3_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel6, 2)  ,  simde_mm_srli_epi16(inPixel7, 2)  );

            //List1

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride + 8));
            inPixel6 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride + 16));
            inPixel7 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride + 24));

           	out8_2_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
            out8_3_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel6, 2)  ,  simde_mm_srli_epi16(inPixel7, 2)  );
            
            //AVG
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);
            avg8_3_U8 =  simde_mm_avg_epu8 (out8_3_U8_L0 , out8_3_U8_L1);         
#if ALSTORE
            simde_mm_store_si128((simde__m128i*)(dstPtr  + dstStride      ) , avg8_2_U8);
			simde_mm_store_si128((simde__m128i*)(dstPtr  + dstStride + 16 ) , avg8_3_U8);
#else
			simde_mm_storeu_si128((simde__m128i*)(dstPtr  + dstStride      ) , avg8_2_U8);
			simde_mm_storeu_si128((simde__m128i*)(dstPtr  + dstStride + 16 ) , avg8_3_U8);
#endif         	
      
#endif            
            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;

        }

	}
	else if (width == 64)
	{
       

#if B256
        simde__m256i inVal16b_0,inVal16b_1,inVal16b_2,inVal16b_3;
        simde__m256i data8b_32_0_L0,data8b_32_1_L0,data8b_32_0_L1,data8b_32_1_L1;
        simde__m256i avg8b_32_0,avg8b_32_1;
#else 
        simde__m128i inPixel2, inPixel3, inPixel4, inPixel5, inPixel6, inPixel7;
        simde__m128i out8_0_U8_L0, out8_1_U8_L0, out8_2_U8_L0, out8_3_U8_L0;
        simde__m128i out8_0_U8_L1, out8_1_U8_L1, out8_2_U8_L1, out8_3_U8_L1;
        simde__m128i avg8_0_U8, avg8_1_U8, avg8_2_U8, avg8_3_U8;

#endif

		for (y = 0; y < height; ++y)
		{

#if B256       // simde_mm256_lddqu_si256

            //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L0);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 16));
            inVal16b_2 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 32));
            inVal16b_3 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 48));
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
            data8b_32_1_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_2, 2) ,  simde_mm256_srli_epi16 (inVal16b_3, 2));
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L1);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 16));
            inVal16b_2 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 32));
            inVal16b_3 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 48));
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
            data8b_32_1_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_2, 2) ,  simde_mm256_srli_epi16 (inVal16b_3, 2));
                       
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);
            avg8b_32_1 = simde_mm256_avg_epu8(data8b_32_1_L0,data8b_32_1_L1);

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);
            avg8b_32_1 = simde_mm256_permute4x64_epi64(avg8b_32_1, 216);
           
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr     ), avg8b_32_0);
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr + 32), avg8b_32_1);
#else
            //List0
            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L0);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 8));
            inPixel2 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 16));
            inPixel3 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 24));
            inPixel4 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 32));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 40));
            inPixel6 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 48));
            inPixel7 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + 56));
            			

            out8_0_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
			out8_1_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel2, 2)  ,  simde_mm_srli_epi16(inPixel3, 2)  );
			out8_2_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
            out8_3_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel6, 2)  ,  simde_mm_srli_epi16(inPixel7, 2)  );


           
		
            //List1
            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 8));
            inPixel2 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 16));
            inPixel3 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 24));
            inPixel4 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 32));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 40));
            inPixel6 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 48));
            inPixel7 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 56));


            //Note: old Version used to use simde_mm_and_si128 to mask the MSB bits of the pixels 
            out8_0_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
			out8_1_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel2, 2)  ,  simde_mm_srli_epi16(inPixel3, 2)  );
			out8_2_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
            out8_3_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel6, 2)  ,  simde_mm_srli_epi16(inPixel7, 2)  );

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);
            avg8_1_U8 =  simde_mm_avg_epu8 (out8_1_U8_L0 , out8_1_U8_L1);
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);
            avg8_3_U8 =  simde_mm_avg_epu8 (out8_3_U8_L0 , out8_3_U8_L1);
#if ALSTORE
            simde_mm_store_si128((simde__m128i*) dstPtr      , avg8_0_U8);
			simde_mm_store_si128((simde__m128i*)(dstPtr + 16), avg8_1_U8);
			simde_mm_store_si128((simde__m128i*)(dstPtr + 32), avg8_2_U8);
			simde_mm_store_si128((simde__m128i*)(dstPtr + 48), avg8_3_U8);
#else
			simde_mm_storeu_si128((simde__m128i*) dstPtr      , avg8_0_U8);
			simde_mm_storeu_si128((simde__m128i*)(dstPtr + 16), avg8_1_U8);
			simde_mm_storeu_si128((simde__m128i*)(dstPtr + 32), avg8_2_U8);
			simde_mm_storeu_si128((simde__m128i*)(dstPtr + 48), avg8_3_U8);			
#endif
	
#endif
			dstPtr  += dstStride;
            ref16L0 += refL0Stride;
			ref16L1 += refL1Stride;
		}
	}
	

    return;
}

EB_S32  sumResidual8bit_AVX2_INTRIN(
                     EB_S16 * inPtr,
                     EB_U32   size,
                     EB_U32   strideIn )
{

   EB_S32  sumBlock;

   simde__m128i in0,in1, in01,in2,in3,in23,sum,sumL,sumH;
   simde__m256i sum0,sum1, sum2, sum3, sum0L,sum0H,sumT,sum01,sumTPerm;
   EB_U32 rowIndex;

   //Assumption: 9bit or 11bit residual data . for bigger block sizes or bigger bit depths , re-asses the dynamic range of the internal calculation

   if(size==4){ //SSSE3
       
        simde__m128i zer = simde_mm_setzero_si128();

       in0  = simde_mm_loadl_epi64((simde__m128i*)inPtr);
       in1  = simde_mm_loadl_epi64((simde__m128i*)(inPtr+strideIn));
       in1  = simde_mm_shuffle_epi32 (in1, 0x4A ); //01.00.10.10
       in01 = simde_mm_or_si128 (in1, in0);

       in2  = simde_mm_loadl_epi64((simde__m128i*)(inPtr+2*strideIn));
       in3  = simde_mm_loadl_epi64((simde__m128i*)(inPtr+3*strideIn));
       in3  = simde_mm_shuffle_epi32 (in3, 0x4A ); //01.00.10.10
       in23 = simde_mm_or_si128 (in3, in2);

       sum  = simde_mm_add_epi16 (in01, in23);
       sum  = simde_mm_hadd_epi16 (sum, zer);
       sum  = simde_mm_hadd_epi16 (sum, zer);
       sum  = simde_mm_hadd_epi16 (sum, zer);

       sum      = simde_mm_cvtepi16_epi32(sum);
       sumBlock = simde_mm_cvtsi128_si32 (sum);

       return sumBlock;
   
   }else if(size==8){//SSSE3
   
        simde__m128i zer = simde_mm_setzero_si128();

        sum  =  simde_mm_add_epi16 (simde_mm_loadu_si128((simde__m128i*)(inPtr+0*strideIn)),  simde_mm_loadu_si128((simde__m128i*)(inPtr+1*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+2*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+3*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+4*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+5*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+6*strideIn)));
        sum  =  simde_mm_add_epi16 (sum, simde_mm_loadu_si128((simde__m128i*)(inPtr+7*strideIn)));

        sum  = simde_mm_hadd_epi16 (sum, zer);
        sum  = simde_mm_hadd_epi16 (sum, zer);
        sum  = simde_mm_hadd_epi16 (sum, zer);

        sum      = simde_mm_cvtepi16_epi32(sum); //the sum is on 16bit, for negative values, we need to extend the sign to the next 16bit, so that the next extraction to int is fine.
        sumBlock = simde_mm_cvtsi128_si32 (sum);

        return sumBlock;
   
   }else if(size==16){//AVX2

         sum0  =  simde_mm256_add_epi16 (simde_mm256_loadu_si256((simde__m256i *)(inPtr+0*strideIn)),  simde_mm256_loadu_si256((simde__m256i *)(inPtr+1*strideIn)));
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+2*strideIn)));
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+3*strideIn))); 
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+4*strideIn)));
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+5*strideIn)));
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+6*strideIn)));
         sum0  =  simde_mm256_add_epi16 (sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtr+7*strideIn)));

         inPtr+=8*strideIn;
         sum1  =  simde_mm256_add_epi16 (simde_mm256_loadu_si256((simde__m256i *)(inPtr+0*strideIn)),  simde_mm256_loadu_si256((simde__m256i *)(inPtr+1*strideIn)));
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+2*strideIn)));
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+3*strideIn))); 
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+4*strideIn)));
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+5*strideIn)));
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+6*strideIn)));
         sum1  =  simde_mm256_add_epi16 (sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtr+7*strideIn)));

         sum01  =  simde_mm256_add_epi16 (sum0,sum1);       
  
         //go from 16bit to 32bit (to support big values)
         sumL   = simde_mm256_castsi256_si128(sum01);
         sumH   = simde_mm256_extracti128_si256(sum01,1);
         sum0L  = simde_mm256_cvtepi16_epi32(sumL);
         sum0H  = simde_mm256_cvtepi16_epi32(sumH);

         sumT     = simde_mm256_add_epi32(sum0L,sum0H); 

         sumT     = simde_mm256_hadd_epi32(sumT,sumT); 
         sumT     = simde_mm256_hadd_epi32(sumT,sumT); 
         sumTPerm = simde_mm256_permute4x64_epi64(sumT, 2); //00.00.00.10       
         sumT     = simde_mm256_add_epi32(sumT,sumTPerm);

         sum      = simde_mm256_castsi256_si128(sumT); 
         sumBlock = simde_mm_cvtsi128_si32 (sum);  

         return sumBlock;

   }
   else if (size == 32){//AVX2
       EB_S16 *inPtrTemp = inPtr;

       sum0 = sum1 = sum2 = sum3 = simde_mm256_setzero_si256();
       for (rowIndex = 0; rowIndex < size; rowIndex += 2) { // Parse every two rows
           sum0 = simde_mm256_add_epi16(sum0, simde_mm256_loadu_si256((simde__m256i *)(inPtrTemp)));
           sum1 = simde_mm256_add_epi16(sum1, simde_mm256_loadu_si256((simde__m256i *)(inPtrTemp + 16)));
           inPtrTemp += strideIn;
           sum2 = simde_mm256_add_epi16(sum2, simde_mm256_loadu_si256((simde__m256i *)(inPtrTemp)));
           sum3 = simde_mm256_add_epi16(sum3, simde_mm256_loadu_si256((simde__m256i *)(inPtrTemp + 16)));
           inPtrTemp += strideIn;

       }
       //go from 16bit to 32bit (to support big values)
       sumL = simde_mm256_castsi256_si128(sum0);
       sumH = simde_mm256_extracti128_si256(sum0, 1);
       sum0L = simde_mm256_cvtepi16_epi32(sumL);
       sum0H = simde_mm256_cvtepi16_epi32(sumH);
       sumT = simde_mm256_add_epi32(sum0L, sum0H);

       sumL = simde_mm256_castsi256_si128(sum1);
       sumH = simde_mm256_extracti128_si256(sum1, 1);
       sum0L = simde_mm256_cvtepi16_epi32(sumL);
       sum0H = simde_mm256_cvtepi16_epi32(sumH);
       sumT = simde_mm256_add_epi32(sumT, sum0L);
       sumT = simde_mm256_add_epi32(sumT, sum0H);

       sumL = simde_mm256_castsi256_si128(sum2);
       sumH = simde_mm256_extracti128_si256(sum2, 1);
       sum0L = simde_mm256_cvtepi16_epi32(sumL);
       sum0H = simde_mm256_cvtepi16_epi32(sumH);
       sumT = simde_mm256_add_epi32(sumT, sum0L);
       sumT = simde_mm256_add_epi32(sumT, sum0H);

       sumL = simde_mm256_castsi256_si128(sum3);
       sumH = simde_mm256_extracti128_si256(sum3, 1);
       sum0L = simde_mm256_cvtepi16_epi32(sumL);
       sum0H = simde_mm256_cvtepi16_epi32(sumH);
       sumT = simde_mm256_add_epi32(sumT, sum0L);
       sumT = simde_mm256_add_epi32(sumT, sum0H);

       sumT = simde_mm256_hadd_epi32(sumT, sumT);
       sumT = simde_mm256_hadd_epi32(sumT, sumT);
       sumTPerm = simde_mm256_permute4x64_epi64(sumT, 2); //00.00.00.10       
       sumT = simde_mm256_add_epi32(sumT, sumTPerm);

       sum = simde_mm256_castsi256_si128(sumT);
       sumBlock = simde_mm_cvtsi128_si32(sum);

       return sumBlock;
   }

   else{
       return 0;
   }


}

void memset16bitBlock_AVX2_INTRIN (
                    EB_S16 * inPtr,
                    EB_U32   strideIn,
                    EB_U32   size,
                    EB_S16   value
    )
{


   if(size==4){ 

        simde__m128i line =  simde_mm_set1_epi16 (value); 

        simde_mm_storel_epi64 ((simde__m128i *)(inPtr + 0*strideIn), line);
        simde_mm_storel_epi64 ((simde__m128i *)(inPtr + 1*strideIn), line);
        simde_mm_storel_epi64 ((simde__m128i *)(inPtr + 2*strideIn), line);
        simde_mm_storel_epi64 ((simde__m128i *)(inPtr + 3*strideIn), line);
   
   }else if(size==8){
   
       simde__m128i line =  simde_mm_set1_epi16 (value); 

        simde_mm_storeu_si128((simde__m128i *)(inPtr + 0*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 1*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 2*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 3*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 4*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 5*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 6*strideIn), line);
        simde_mm_storeu_si128((simde__m128i *)(inPtr + 7*strideIn), line);
   
   }else if(size==16){

       simde__m256i line =  simde_mm256_set1_epi16(value); 

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7*strideIn), line);

       inPtr+=8*strideIn;

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6*strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7*strideIn), line);


   }
   else if (size == 32){

       simde__m256i line = simde_mm256_set1_epi16(value);

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn + 16), line);

       inPtr += 8 * strideIn;

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn + 16), line);

       inPtr += 8 * strideIn;

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn + 16), line);

       inPtr += 8 * strideIn;

       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 0 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 1 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 2 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 3 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 4 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 5 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 6 * strideIn + 16), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn), line);
       simde_mm256_storeu_si256((simde__m256i *)(inPtr + 7 * strideIn + 16), line);

   }


}

void UnpackAvgSafeSub_AVX2_INTRIN(
	EB_U16 *ref16L0,
        EB_U32  refL0Stride,
        EB_U16 *ref16L1,
        EB_U32  refL1Stride,
        EB_U8  *dstPtr,
        EB_U32  dstStride,  
        EB_U32  width,
        EB_U32  height)
{

	EB_U32   y;
    simde__m128i inPixel0, inPixel1;  

  
    if (width == 8)
	{
		
        simde__m128i out8_0_U8_L0, out8_0_U8_L1, out8_2_U8_L0,out8_2_U8_L1;
        simde__m128i avg8_0_U8,avg8_2_U8;
        
        for (y = 0; y < height; y += 2)
		{
            //--------
            //Line One
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L0); 

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_0_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

             //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;
            out8_0_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           

			simde_mm_storel_epi64((simde__m128i*) dstPtr      , avg8_0_U8);		

            //--------
            //Line Two
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)(ref16L0 + refL0Stride) );

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;         
            out8_2_U8_L0 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + refL1Stride) );

            inPixel1 = simde_mm_srli_epi16(inPixel0, 2) ;
            out8_2_U8_L1 = simde_mm_packus_epi16( inPixel1 , inPixel1 );		

            //AVG
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);           

			simde_mm_storel_epi64((simde__m128i*)(dstPtr +dstStride)    , avg8_2_U8);

            	
            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;
        }
		
	}
	else if (width == 16)
	{

        simde__m128i inPixel4, inPixel5;
        simde__m128i out8_0_U8_L0, out8_0_U8_L1, out8_2_U8_L0,out8_2_U8_L1;
        simde__m128i avg8_0_U8,avg8_2_U8;

        for (y = 0; y < height; y += 2)
		{

            //--------
            //Line One
            //--------

            //List0

            inPixel0 = simde_mm_loadu_si128((simde__m128i*)  ref16L0);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + 8));
      
            out8_0_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );		

             //List1

            inPixel0 = simde_mm_loadu_si128((simde__m128i*) ref16L1);
            inPixel1 = simde_mm_loadu_si128((simde__m128i*)(ref16L1 + 8));          
    
            out8_0_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel0, 2)  ,  simde_mm_srli_epi16(inPixel1, 2)  );
	

            //AVG
            avg8_0_U8 =  simde_mm_avg_epu8 (out8_0_U8_L0 , out8_0_U8_L1);           

            simde_mm_store_si128((simde__m128i*) dstPtr      , avg8_0_U8);

            //--------
            //Line Two
            //--------

            //List0

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L0 + refL0Stride + 8));         

           	out8_2_U8_L0 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
          
            //List1

            inPixel4 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride));
            inPixel5 = simde_mm_loadu_si128((simde__m128i*) (ref16L1 + refL1Stride + 8));          

           	out8_2_U8_L1 = simde_mm_packus_epi16(  simde_mm_srli_epi16(inPixel4, 2)  ,  simde_mm_srli_epi16(inPixel5, 2)  );
         
            
            //AVG
            avg8_2_U8 =  simde_mm_avg_epu8 (out8_2_U8_L0 , out8_2_U8_L1);               

            simde_mm_store_si128((simde__m128i*)(dstPtr  + dstStride      ) , avg8_2_U8);	

            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;

        }

	}
	else if (width == 32)
	{

        simde__m256i inVal16b_0,inVal16b_1;
        simde__m256i data8b_32_0_L0,data8b_32_0_L1;
        simde__m256i avg8b_32_0;      

       for (y = 0; y < height; y += 2)
		{
            
            //--------
            //Line One
            //--------

            //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L0);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 16));         
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));            
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L1);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 16));          
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
                             
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);          

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);
                     
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr     ), avg8b_32_0);
           
            //--------
            //Line Two
            //--------
              //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + refL0Stride     ));
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + refL0Stride + 16));
         
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
         
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + refL1Stride     ));
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + refL1Stride + 16));
           
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
                                
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);           

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);          
           
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr + dstStride   ), avg8b_32_0);
           
          
            dstPtr  += 2*dstStride;
            ref16L0 += 2*refL0Stride;
			ref16L1 += 2*refL1Stride;

        }


	}
	else if (width == 64)
	{
        simde__m256i inVal16b_0,inVal16b_1,inVal16b_2,inVal16b_3;
        simde__m256i data8b_32_0_L0,data8b_32_1_L0,data8b_32_0_L1,data8b_32_1_L1;
        simde__m256i avg8b_32_0,avg8b_32_1;


		for (y = 0; y < height; ++y)
		{


            //List0
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L0);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 16));
            inVal16b_2 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 32));
            inVal16b_3 = simde_mm256_loadu_si256((simde__m256i*)(ref16L0 + 48));
            data8b_32_0_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
            data8b_32_1_L0 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_2, 2) ,  simde_mm256_srli_epi16 (inVal16b_3, 2));
            //List1
            inVal16b_0 = simde_mm256_loadu_si256((simde__m256i*) ref16L1);
            inVal16b_1 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 16));
            inVal16b_2 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 32));
            inVal16b_3 = simde_mm256_loadu_si256((simde__m256i*)(ref16L1 + 48));
            data8b_32_0_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_0, 2) ,  simde_mm256_srli_epi16 (inVal16b_1, 2));
            data8b_32_1_L1 = simde_mm256_packus_epi16(  simde_mm256_srli_epi16 (inVal16b_2, 2) ,  simde_mm256_srli_epi16 (inVal16b_3, 2));
                       
           
            //Avg
            avg8b_32_0 = simde_mm256_avg_epu8(data8b_32_0_L0,data8b_32_0_L1);
            avg8b_32_1 = simde_mm256_avg_epu8(data8b_32_1_L0,data8b_32_1_L1);

            avg8b_32_0 = simde_mm256_permute4x64_epi64(avg8b_32_0, 216);
            avg8b_32_1 = simde_mm256_permute4x64_epi64(avg8b_32_1, 216);
           
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr     ), avg8b_32_0);
            simde_mm256_storeu_si256((simde__m256i *)(dstPtr + 32), avg8b_32_1);

			dstPtr  += dstStride;
            ref16L0 += refL0Stride;
			ref16L1 += refL1Stride;
		}


	}
	

    return;
}

