#include <iostream>
#include <cmath>
#include <cstdint>
#include "logistic_fault.h"

static int shift_for_window(int window_size) {
    if (window_size == 256) return 8;
    if (window_size == 512) return 9;
    if (window_size == 1024) return 10;
    return 11;
}


// Encode an ORIGINAL, unscaled CWRU-unit sample into the signed raw ap_fixed
// bit code. The testbench stores codes in int16_t; HLS reconstructs the
// binary point. No amplitude normalization multiplier is used.
static int16_t encode_fixed_sample(double value) {
    const long long scale =
        (1LL << SAMPLE_FRAC_BITS);

    long long code =
        (long long)std::llround(
            value * (double)scale
        );

    const long long code_min =
        -(1LL << (SAMPLE_BITS - 1));

    const long long code_max =
        (1LL << (SAMPLE_BITS - 1)) - 1;

    if (code < code_min) code = code_min;
    if (code > code_max) code = code_max;

    return (int16_t)code;
}


static bus_word_t pack_bus_word(
    const int16_t *window_codes,
    int window_size,
    int base_sample,
    int bus
) {
    bus_word_t word = 0;

PACK_BUS_LOOP:
    for (int lane = 0; lane < SAMPLES_PER_BUS; lane++) {
        int sample_idx =
            base_sample +
            bus * SAMPLES_PER_BUS +
            lane;

        ap_int<SAMPLE_BITS> code = 0;

        if (sample_idx < window_size) {
            code =
                (ap_int<SAMPLE_BITS>)
                window_codes[sample_idx];
        }

        // Pack the raw two's-complement fixed-point code bits.
        word.range(
            lane * SAMPLE_BITS + SAMPLE_BITS - 1,
            lane * SAMPLE_BITS
        ) = (ap_uint<SAMPLE_BITS>)code;
    }

    return word;
}


static void stream_windows_four_bus(
    const int16_t raw_codes[][MAX_WINDOW_SIZE],
    int test_windows,
    int window_size,
    hls::stream<axis_bus_word_t> &input_stream0,
    hls::stream<axis_bus_word_t> &input_stream1,
    hls::stream<axis_bus_word_t> &input_stream2,
    hls::stream<axis_bus_word_t> &input_stream3
) {
    const int packets_per_window =
        (window_size + SAMPLES_PER_PACKET - 1) /
        SAMPLES_PER_PACKET;

    const int total_packets =
        test_windows * packets_per_window;

    int sent = 0;

    for (int w = 0; w < test_windows; w++) {
        for (int p = 0; p < packets_per_window; p++) {
            const int base =
                p * SAMPLES_PER_PACKET;

            axis_bus_word_t a0, a1, a2, a3;

            a0.data = pack_bus_word(raw_codes[w], window_size, base, 0);
            a1.data = pack_bus_word(raw_codes[w], window_size, base, 1);
            a2.data = pack_bus_word(raw_codes[w], window_size, base, 2);
            a3.data = pack_bus_word(raw_codes[w], window_size, base, 3);

            a0.keep = -1; a1.keep = -1;
            a2.keep = -1; a3.keep = -1;

            a0.strb = -1; a1.strb = -1;
            a2.strb = -1; a3.strb = -1;

            bool is_last =
                (sent == total_packets - 1);

            a0.last = is_last; a1.last = is_last;
            a2.last = is_last; a3.last = is_last;

            input_stream0.write(a0);
            input_stream1.write(a1);
            input_stream2.write(a2);
            input_stream3.write(a3);

            sent++;
        }
    }
}


// Software reference that exactly mirrors the fixed-point feature definitions.
// Peak and MAV are decoded with SAMPLE_FRAC_BITS. MAV divides by N using the
// full right shift (>> log2(N)); no fractional-bit subtraction is used.
static void reference_features(
    const int16_t *w_code,
    int window_size,
    double out[NUM_FEATURES]
) {
    const long long scale =
        (1LL << SAMPLE_FRAC_BITS);

    int peak_code =
        -(1 << (SAMPLE_BITS - 1));

    long long mav_sum_code = 0;
    int zc = 0;

    double sample_sum = 0.0;

    for (int i = 0; i < window_size; i++) {
        int x_code =
            (int)w_code[i];

        if (x_code > peak_code)
            peak_code = x_code;

        mav_sum_code +=
            (x_code < 0)
            ? -(long long)x_code
            :  (long long)x_code;

        sample_sum +=
            (double)x_code /
            (double)scale;

        if (i > 0) {
            int a_code =
                (int)w_code[i - 1];

            int b_code =
                x_code;

            bool sign_change =
                ((a_code < 0) && (b_code > 0)) ||
                ((a_code > 0) && (b_code < 0));

            long long delta_code =
                (long long)b_code -
                (long long)a_code;

            if (delta_code < 0)
                delta_code = -delta_code;

            if (
                sign_change &&
                delta_code > ZC_EPSILON_CODE
            ) {
                zc++;
            }
        }
    }

    const int shift =
        shift_for_window(window_size);

    long long mav_code =
        mav_sum_code >> shift;

    const double mean =
        sample_sum /
        (double)window_size;

    double centered2_sum = 0.0;
    double centered3_sum = 0.0;
    double centered4_sum = 0.0;

    for (int i = 0; i < window_size; i++) {
        const double value =
            (double)w_code[i] /
            (double)scale;

        const double centered =
            value - mean;

        const double centered2 =
            centered * centered;

        centered2_sum +=
            centered2;

        centered3_sum +=
            centered2 * centered;

        centered4_sum +=
            centered2 * centered2;
    }

    const double variance =
        centered2_sum /
        (double)window_size;

    double kurtosis = 0.0;
    double normal_nll = 0.0;
    double skewness = 0.0;

    if (variance >= 1e-12) {
        const double stddev =
            std::sqrt(variance);

        skewness =
            (
                centered3_sum /
                (double)window_size
            ) /
            (
                stddev *
                stddev *
                stddev
            );

        kurtosis =
            (
                centered4_sum /
                (double)window_size
            ) /
            (
                variance *
                variance
            );

        const double TWO_PI =
            6.28318530717958647692;

        normal_nll =
            0.5 *
            (double)window_size *
            std::log(
                TWO_PI *
                variance
            )
            +
            centered2_sum /
            (
                2.0 *
                variance
            );
    }

    // Match the final HLS statistical feature output representation.
    stat_feature_t kurt_q =
        (stat_feature_t)kurtosis;

    stat_feature_t nll_q =
        (stat_feature_t)normal_nll;

    stat_feature_t skew_q =
        (stat_feature_t)skewness;

    out[0] =
        (double)peak_code /
        (double)scale;

    out[1] =
        (double)mav_code /
        (double)scale;

    out[2] =
        (double)zc;

    out[3] =
        (double)kurt_q;

    out[4] =
        (double)nll_q;

    out[5] =
        (double)skew_q;
}


static int reference_class(
    const double feat[NUM_FEATURES]
) {
    feature_t peak =
        (feature_t)feat[0];

    feature_t mav =
        (feature_t)feat[1];

    feature_t zc =
        (feature_t)feat[2];

    stat_feature_t kurtosis =
        (stat_feature_t)feat[3];

    stat_feature_t normal_nll =
        (stat_feature_t)feat[4];

    stat_feature_t skewness =
        (stat_feature_t)feat[5];

    acc_t score[NUM_CLASSES];

    for (int c = 0; c < NUM_CLASSES; c++) {
        score[c] =
            (acc_t)LR_HW_BIAS[c];

        score[c] +=
            (acc_t)(
                peak *
                LR_PEAK_WEIGHTS[c]
            );

        score[c] +=
            (acc_t)(
                mav *
                LR_MAV_WEIGHTS[c]
            );

        score[c] +=
            (acc_t)(
                zc *
                LR_ZC_WEIGHTS[c]
            );

        score[c] +=
            (acc_t)(
                kurtosis *
                LR_KURTOSIS_WEIGHTS[c]
            );

        score[c] +=
            (acc_t)(
                normal_nll *
                LR_NLL_WEIGHTS[c]
            );

        score[c] +=
            (acc_t)(
                skewness *
                LR_SKEWNESS_WEIGHTS[c]
            );
    }

    int best = 0;

    for (int c = 1; c < NUM_CLASSES; c++) {
        if (score[c] > score[best])
            best = c;
    }

    return best;
}


static void fill_test_window(
    int16_t *window_codes,
    int window_size,
    int pattern
) {
    for (int i = 0; i < window_size; i++) {
        double value = 0.0;

        switch (pattern & 7) {
        case 0:
            value =
                (i & 1) ? 0.5 : -0.5;
            break;
        case 1:
            value = 6.0;
            break;
        case 2:
            value =
                (i & 1) ? 0.25 : -0.25;
            break;
        case 3:
            value =
                ((i % 17) - 8) / 8.0;
            break;
        case 4:
            value =
                ((i / 8) & 1)
                ? 6.5
                : -6.5;
            break;
        case 5:
            value =
                (i % 5 == 0)
                ? 1.5
                : -0.125;
            break;
        case 6:
            value =
                (((i * 7) % 25) - 12) / 8.0;
            break;
        default:
            value =
                (((i * 3) % 17) - 8) / 8.0;
            break;
        }

        window_codes[i] =
            encode_fixed_sample(value);
    }
}


static bool run_window_size_test(
    int window_size
) {
    const int TEST_WINDOWS = 8;

    static int16_t
        raw_codes[TEST_WINDOWS][MAX_WINDOW_SIZE];

    static int
        predictions[TEST_WINDOWS];

    hls::stream<axis_bus_word_t>
        input_stream0("input_stream0");

    hls::stream<axis_bus_word_t>
        input_stream1("input_stream1");

    hls::stream<axis_bus_word_t>
        input_stream2("input_stream2");

    hls::stream<axis_bus_word_t>
        input_stream3("input_stream3");

    for (int w = 0; w < TEST_WINDOWS; w++)
        fill_test_window(
            raw_codes[w],
            window_size,
            w
        );

    stream_windows_four_bus(
        raw_codes,
        TEST_WINDOWS,
        window_size,
        input_stream0,
        input_stream1,
        input_stream2,
        input_stream3
    );

    logistic_fault_predict(
        input_stream0,
        input_stream1,
        input_stream2,
        input_stream3,
        predictions,
        TEST_WINDOWS,
        window_size
    );

    bool ok = true;

    std::cout
        << "\nSAMPLE_BITS=" << SAMPLE_BITS
        << " SAMPLE_INT_BITS=" << SAMPLE_INT_BITS
        << " SAMPLE_FRAC_BITS=" << SAMPLE_FRAC_BITS
        << " SAMPLES_PER_BUS=" << SAMPLES_PER_BUS
        << " SAMPLES_PER_PACKET=" << SAMPLES_PER_PACKET
        << " Window size " << window_size
        << " Packets/window "
        << (
            (window_size + SAMPLES_PER_PACKET - 1) /
            SAMPLES_PER_PACKET
        )
        << "\n";

    for (int w = 0; w < TEST_WINDOWS; w++) {
        double feat[NUM_FEATURES];

        reference_features(
            raw_codes[w],
            window_size,
            feat
        );

        int ref =
            reference_class(feat);

        std::cout
            << "  W" << w
            << " features=["
            << feat[0] << ", "
            << feat[1] << ", "
            << feat[2] << ", "
            << feat[3] << ", "
            << feat[4] << ", "
            << feat[5] << "]"
            << " HW=" << predictions[w]
            << " REF=" << ref
            << "\n";

        if (predictions[w] != ref)
            ok = false;
    }

    return ok;
}


int main() {
    bool ok =
        run_window_size_test(
            MODEL_WINDOW_SIZE
        );

    if (!ok) {
        std::cerr
            << "\nTEST FAILED\n";
        return 1;
    }

    std::cout
        << "\nTEST PASSED\n";

    return 0;
}
