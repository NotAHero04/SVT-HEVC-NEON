/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "../../../simde/simde/x86/avx2.h"

#include "EbPackUnPack_Intrinsic_AVX2.h"
#include "EbDefinitions.h"

#ifndef NON_AVX512_SUPPORT
AVX512_FUNC_TARGET
void EB_ENC_msbUnPack2D_AVX512_INTRIN(
    EB_U16      *in16BitBuffer,
    EB_U32       inStride,
    EB_U8       *out8BitBuffer,
    EB_U8       *outnBitBuffer,
    EB_U32       out8Stride,
    EB_U32       outnStride,
    EB_U32       width,
    EB_U32       height)
{
    const EB_U32 CHUNK_SIZE = 64;
    simde__m512i zmm_3 = simde_mm512_set1_epi16(0x0003);
    simde__m512i zmm_perm = simde_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    EB_U32 x, y;

    EB_U32 out8BitBufferAlign = CHUNK_SIZE - ((size_t)out8BitBuffer & (CHUNK_SIZE - 1));
    EB_U32 outnBitBufferAlign = CHUNK_SIZE - ((size_t)outnBitBuffer & (CHUNK_SIZE - 1));

    // We are going to use non temporal stores for both output buffers if they are aligned or can be aligned using same offset
    const int b_use_nt_store_for_both = (out8BitBufferAlign == outnBitBufferAlign) && (out8Stride == outnStride);

    for (x = 0; x < height; x++) {
            // Find offset to the aligned store address
            EB_U32 offset = (CHUNK_SIZE - (size_t)out8BitBuffer) & (CHUNK_SIZE - 1);
            EB_U32 complOffset = (width - offset) & (CHUNK_SIZE - 1);
            EB_U32 numChunks = (width - (offset + complOffset)) / CHUNK_SIZE;

            // Process the unaligned output data pixel by pixel
            for (y = 0; y < offset; y++)
            {
                EB_U16 inPixel0 = in16BitBuffer[y];
                out8BitBuffer[y] = (EB_U8)(inPixel0 >> 2);

                EB_U8 tmpPixel0 = (EB_U8)(inPixel0 << 6);
                outnBitBuffer[y] = tmpPixel0;
            }
            in16BitBuffer += offset;
            outnBitBuffer += offset;
            out8BitBuffer += offset;

            for (y = 0; y < numChunks; y++) {
                // We process scan lines by elements of CHUNK_SIZE starting from an offset and after that we process the residual (or complimentary offset) going pixel by pixel

                simde__m512i inPixel0 = simde_mm512_loadu_si512((simde__m512i*)in16BitBuffer);
                simde__m512i inPixel1 = simde_mm512_loadu_si512((simde__m512i*)(in16BitBuffer + 32));

                simde__m512i outn0_U8 = simde_mm512_slli_epi16(simde_mm512_packus_epi16(simde_mm512_and_si512(inPixel0, zmm_3), simde_mm512_and_si512(inPixel1, zmm_3)), 6);
                simde__m512i out8_0_U8 = simde_mm512_packus_epi16(simde_mm512_srli_epi16(inPixel0, 2), simde_mm512_srli_epi16(inPixel1, 2));

                outn0_U8 = simde_mm512_permutexvar_epi64(zmm_perm, outn0_U8);
                out8_0_U8 = simde_mm512_permutexvar_epi64(zmm_perm, out8_0_U8);

                simde_mm512_stream_si512((simde__m512i*)out8BitBuffer, out8_0_U8);
                if (b_use_nt_store_for_both)
                    simde_mm512_stream_si512((simde__m512i*)outnBitBuffer, outn0_U8);
                else
                    simde_mm512_storeu_si512((simde__m512i*)outnBitBuffer, outn0_U8);
                outnBitBuffer += CHUNK_SIZE;
                out8BitBuffer += CHUNK_SIZE;
                in16BitBuffer += CHUNK_SIZE;
        }

        // Process the tail of a scan line
        for (y = 0; y < complOffset; y++)
        {
            EB_U16 inPixel2 = in16BitBuffer[y];
            out8BitBuffer[y] = (EB_U8)(inPixel2 >> 2);
                
            EB_U8 tmpPixel2 = (EB_U8)(inPixel2 << 6);
            outnBitBuffer[y] = tmpPixel2;
        }

        in16BitBuffer += inStride - width + complOffset;
        outnBitBuffer += outnStride - width + complOffset;
        out8BitBuffer += out8Stride - width + complOffset;
    }

    return;
}
#else
void EB_ENC_msbUnPack2D_AVX2_INTRIN(
    EB_U16      *in16BitBuffer,
    EB_U32       inStride,
    EB_U8       *out8BitBuffer,
    EB_U8       *outnBitBuffer,
    EB_U32       out8Stride,
    EB_U32       outnStride,
    EB_U32       width,
    EB_U32       height)
{
    const EB_U32 CHUNK_SIZE = 32;
    const EB_U32 out8BitBufferAlign = CHUNK_SIZE - ((size_t)out8BitBuffer & (CHUNK_SIZE - 1));
    const EB_U32 outnBitBufferAlign = CHUNK_SIZE - ((size_t)outnBitBuffer & (CHUNK_SIZE - 1));
    const simde__m256i ymm_3 = simde_mm256_set1_epi16(0x0003);
    EB_U32 x, y;

    // We are going to use non temporal stores for both output buffers if they are aligned or can be aligned using same offset
    const int b_use_nt_store_for_both = (out8BitBufferAlign == outnBitBufferAlign) && (out8Stride == outnStride);

    for (x = 0; x < height; x++) {
        // Find offset to the aligned store address
        const EB_U32 offset = (CHUNK_SIZE - (size_t)out8BitBuffer) & (CHUNK_SIZE - 1);
        const EB_U32 complOffset = (width - offset) & (CHUNK_SIZE - 1);
        const EB_U32 numChunks = (width - offset - complOffset) / CHUNK_SIZE;

        // Process the unaligned output data pixel by pixel
        for (y = 0; y < offset; y++)
        {
            EB_U16 inPixel0 = in16BitBuffer[y];
            out8BitBuffer[y] = (EB_U8)(inPixel0 >> 2);

            EB_U8 tmpPixel0 = (EB_U8)(inPixel0 << 6);
            outnBitBuffer[y] = tmpPixel0;
        }
        in16BitBuffer += offset;
        outnBitBuffer += offset;
        out8BitBuffer += offset;

        for (y = 0; y < numChunks; y++) {
            // We process scan lines by elements of CHUNK_SIZE starting from an offset and after that we process the residual (or complimentary offset) going pixel by pixel
            const simde__m256i inPixel0 = simde_mm256_loadu_si256((simde__m256i*)in16BitBuffer);
            const simde__m256i inPixel1 = simde_mm256_loadu_si256((simde__m256i*)(in16BitBuffer + CHUNK_SIZE / 2));
            simde__m256i outn0_U8 = simde_mm256_slli_epi16(simde_mm256_packus_epi16(simde_mm256_and_si256(inPixel0, ymm_3), simde_mm256_and_si256(inPixel1, ymm_3)), 6);
            simde__m256i out8_0_U8 = simde_mm256_packus_epi16(simde_mm256_srli_epi16(inPixel0, 2), simde_mm256_srli_epi16(inPixel1, 2));

            outn0_U8 = simde_mm256_permute4x64_epi64(outn0_U8, 0xD8);
            out8_0_U8 = simde_mm256_permute4x64_epi64(out8_0_U8, 0xD8);

            simde_mm256_stream_si256((simde__m256i*)out8BitBuffer, out8_0_U8);
            if (b_use_nt_store_for_both)
                simde_mm256_stream_si256((simde__m256i*)outnBitBuffer, outn0_U8);
            else
                simde_mm256_storeu_si256((simde__m256i*)outnBitBuffer, outn0_U8);
            outnBitBuffer += CHUNK_SIZE;
            out8BitBuffer += CHUNK_SIZE;
            in16BitBuffer += CHUNK_SIZE;
        }

        // Process the tail of a scan line
        for (y = 0; y < complOffset; y++)
        {
            EB_U16 inPixel2 = in16BitBuffer[y];
            out8BitBuffer[y] = (EB_U8)(inPixel2 >> 2);

            EB_U8 tmpPixel2 = (EB_U8)(inPixel2 << 6);
            outnBitBuffer[y] = tmpPixel2;
        }

        in16BitBuffer += inStride - width + complOffset;
        outnBitBuffer += outnStride - width + complOffset;
        out8BitBuffer += out8Stride - width + complOffset;
    }

    return;
}
#endif

