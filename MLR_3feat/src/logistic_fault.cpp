#include "logistic_fault.h"

// ------------------------------------------------------------
// Window configuration helpers
// ------------------------------------------------------------

static bool supported_window_size(int window_size) {
#pragma HLS INLINE
    return (window_size == WINDOW_256) ||
           (window_size == WINDOW_512) ||
           (window_size == WINDOW_1024) ||
           (window_size == WINDOW_2048);
}

static int window_shift(int window_size) {
#pragma HLS INLINE

    if (window_size == WINDOW_256)
        return 8;

    if (window_size == WINDOW_512)
        return 9;

    if (window_size == WINDOW_1024)
        return 10;

    return 11;  // WINDOW_2048
}

static int valid_samples_for_packet(int packet_idx) {
#pragma HLS INLINE

    return (packet_idx == MODEL_PACKETS_PER_WINDOW - 1)
        ? LAST_PACKET_VALID_SAMPLES
        : SAMPLES_PER_PACKET;
}


// ------------------------------------------------------------
// Basic arithmetic helpers
// ------------------------------------------------------------

static magnitude_t abs_internal(raw_t x) {
#pragma HLS INLINE

    internal_t widened = (internal_t)x;

    return (widened < 0)
        ? (magnitude_t)(-widened)
        : (magnitude_t)widened;
}

static bool valid_zero_crossing(
    raw_t a,
    raw_t b
) {
#pragma HLS INLINE

    bool sign_change =
        ((a < 0) && (b > 0)) ||
        ((a > 0) && (b < 0));

    internal_t diff =
        (internal_t)b - (internal_t)a;

    magnitude_t abs_diff =
        (diff < 0)
        ? (magnitude_t)(-diff)
        : (magnitude_t)diff;


    raw_t epsilon = 0;
    epsilon.range(SAMPLE_BITS - 1, 0) =
        (ap_uint<SAMPLE_BITS>)ZC_EPSILON_CODE;

    return sign_change &&
           (abs_diff > (magnitude_t)epsilon);
}

static feature_t mav_from_sum(
    mav_sum_t mav_sum,
    int shift
) {
#pragma HLS INLINE

    mav_sum_t mav_average =
        mav_sum >> shift;

    return (feature_t)mav_average;
}


// ------------------------------------------------------------
// Unpack one 256-bit physical packet into max-packed signed samples.
// ------------------------------------------------------------

static void unpack_packet(
    sample_packet_t packet,
    raw_t x[MAX_SAMPLES_PER_PACKET]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

CLEAR_UNUSED_LANES:
    for (int i = 0; i < MAX_SAMPLES_PER_PACKET; i++) {
#pragma HLS UNROLL
        x[i] = 0;
    }

UNPACK_BUS_LOOP:
    for (int bus = 0; bus < NUM_INPUT_BUSES; bus++) {
#pragma HLS UNROLL

UNPACK_LANE_LOOP:
        for (int lane = 0; lane < SAMPLES_PER_BUS; lane++) {
#pragma HLS UNROLL
            const int sample_idx = bus * SAMPLES_PER_BUS + lane;
            const int lsb = bus * BUS_BITS + lane * SAMPLE_BITS;

            // Bit-copy the SAMPLE_BITS-wide two's-complement fixed-point
            // representation. A numeric cast here would incorrectly treat the
            // packed bits as an integer value before converting to ap_fixed.
            raw_t raw = 0;
            raw.range(SAMPLE_BITS - 1, 0) =
                packet.range(
                    lsb + SAMPLE_BITS - 1,
                    lsb
                );

            x[sample_idx] = raw;
        }
    }
}


// Return how many samples are valid in one 16-lane reduction group.
static int group_valid_samples(
    int valid_samples,
    int base
) {
#pragma HLS INLINE

    int remaining = valid_samples - base;

    if (remaining <= 0)
        return 0;

    return (remaining >= 16) ? 16 : remaining;
}


// Balanced 16-input absolute-value sum. Invalid lanes contribute zero.
typedef ap_fixed<SAMPLE_BITS + 2,SAMPLE_INT_BITS + 2,AP_RND,AP_SAT> abs_sum2_t;

typedef ap_fixed<SAMPLE_BITS + 3,SAMPLE_INT_BITS + 3,AP_RND,AP_SAT> abs_sum4_t;

typedef ap_fixed<SAMPLE_BITS + 4,SAMPLE_INT_BITS + 4,AP_RND,AP_SAT> abs_sum8_t;

typedef ap_fixed<SAMPLE_BITS + 5,SAMPLE_INT_BITS + 5,AP_RND,AP_SAT> abs_sum16_t;

typedef ap_fixed<SAMPLE_BITS + 7,SAMPLE_INT_BITS + 7,AP_RND,AP_SAT> packet_sum_t;


// Balanced 16-input absolute-value sum. Every level keeps the same number of
// fractional bits and adds only the integer width needed by the larger sum.
static abs_sum16_t sum_abs16(
    raw_t x[MAX_SAMPLES_PER_PACKET],
    int base,
    int valid_samples
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

    const int n = group_valid_samples(valid_samples, base);

    magnitude_t a[16];
    abs_sum2_t s1[8];
    abs_sum4_t s2[4];
    abs_sum8_t s3[2];
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s3 complete dim=1

ABS16_INPUT:
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        a[i] = (i < n)
            ? abs_internal(x[base + i])
            : (magnitude_t)0;
    }

ABS16_L1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        s1[i] = (abs_sum2_t)a[2*i] +
                (abs_sum2_t)a[2*i + 1];
    }

ABS16_L2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s2[i] = (abs_sum4_t)s1[2*i] +
                (abs_sum4_t)s1[2*i + 1];
    }

ABS16_L3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s3[i] = (abs_sum8_t)s2[2*i] +
                (abs_sum8_t)s2[2*i + 1];
    }

    return (abs_sum16_t)s3[0] +
           (abs_sum16_t)s3[1];
}


// Balanced 16-input peak. Invalid lanes are filled with the raw minimum so
// zero padding can never become a false peak.
static raw_t peak16(
    raw_t x[MAX_SAMPLES_PER_PACKET],
    int base,
    int valid_samples
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

    const int n = group_valid_samples(valid_samples, base);
    const raw_t RAW_MIN =
        (raw_t)(-(1 << (SAMPLE_INT_BITS - 1)));

    raw_t p0[16];
    raw_t p1[8];
    raw_t p2[4];
    raw_t p3[2];
#pragma HLS ARRAY_PARTITION variable=p0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=p1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=p2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=p3 complete dim=1

PEAK16_INPUT:
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        p0[i] = (i < n) ? x[base + i] : RAW_MIN;
    }

PEAK16_L1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        p1[i] = (p0[2*i] > p0[2*i + 1]) ? p0[2*i] : p0[2*i + 1];
    }

PEAK16_L2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        p2[i] = (p1[2*i] > p1[2*i + 1]) ? p1[2*i] : p1[2*i + 1];
    }

PEAK16_L3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        p3[i] = (p2[2*i] > p2[2*i + 1]) ? p2[2*i] : p2[2*i + 1];
    }

    return (p3[0] > p3[1]) ? p3[0] : p3[1];
}


// Count zero crossings inside one 16-lane group. previous_sample allows the
// first lane in this group to be compared with the preceding valid sample.
static ap_uint<5> zc16(
    raw_t x[MAX_SAMPLES_PER_PACKET],
    int base,
    int valid_samples,
    raw_t previous_sample,
    bool have_previous
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

    const int n = group_valid_samples(valid_samples, base);

    ap_uint<1> c[16];
    ap_uint<2> s1[8];
    ap_uint<3> s2[4];
    ap_uint<4> s3[2];
#pragma HLS ARRAY_PARTITION variable=c complete dim=1
#pragma HLS ARRAY_PARTITION variable=s1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s3 complete dim=1

    c[0] =
        (n > 0 && have_previous &&
         valid_zero_crossing(previous_sample, x[base]))
        ? 1 : 0;

ZC16_FLAGS:
    for (int i = 1; i < 16; i++) {
#pragma HLS UNROLL
        c[i] =
            (i < n && valid_zero_crossing(x[base + i - 1], x[base + i]))
            ? 1 : 0;
    }

ZC16_L1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        s1[i] = (ap_uint<2>)c[2*i] + (ap_uint<2>)c[2*i + 1];
    }

ZC16_L2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s2[i] = (ap_uint<3>)s1[2*i] + (ap_uint<3>)s1[2*i + 1];
    }

ZC16_L3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s3[i] = (ap_uint<4>)s2[2*i] + (ap_uint<4>)s2[2*i + 1];
    }

    return (ap_uint<5>)s3[0] + (ap_uint<5>)s3[1];
}


// All four reads belong to the same logical packet.
//
// Packet ordering:
//   [63:0]    = bus0 (SAMPLES_PER_BUS packed samples)
//   [127:64]  = bus1
//   [191:128] = bus2
//   [255:192] = bus3
//
// Each bus is packed as densely as SAMPLE_BITS allows. For 12-bit samples,
// five samples use 60 bits and the upper four bits remain padding.
static void read_four_streams(
    hls::stream<axis_bus_word_t> &input_stream0,
    hls::stream<axis_bus_word_t> &input_stream1,
    hls::stream<axis_bus_word_t> &input_stream2,
    hls::stream<axis_bus_word_t> &input_stream3,
    hls::stream<sample_packet_t> &packet_stream,
    int total_packets
) {
#pragma HLS INLINE off

READ_FOUR_STREAMS_LOOP:
    for (int idx = 0; idx < total_packets; idx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=1228800

        axis_bus_word_t w0 = input_stream0.read();
        axis_bus_word_t w1 = input_stream1.read();
        axis_bus_word_t w2 = input_stream2.read();
        axis_bus_word_t w3 = input_stream3.read();

        sample_packet_t packet = 0;

        packet.range(63, 0)    = w0.data;
        packet.range(127, 64)  = w1.data;
        packet.range(191, 128) = w2.data;
        packet.range(255, 192) = w3.data;

        packet_stream.write(packet);
    }
}


// The logical packet contains 16, 20, 32, or 64 samples depending on
// SAMPLE_BITS. Reduction is performed as up to four parallel 16-lane trees.
static void mav_task(
    hls::stream<sample_packet_t> &input_stream,
    hls::stream<sample_packet_t> &output_stream,
    hls::stream<feature_t> &mav_result_stream,
    int total_packets,
    int mav_shift
) {
#pragma HLS INLINE off

    mav_sum_t mav_sum = 0;
    int packet_idx = 0;

MAV_PACKET_LOOP:
    for (int idx = 0; idx < total_packets; idx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=1228800

        sample_packet_t packet = input_stream.read();

        raw_t x[MAX_SAMPLES_PER_PACKET];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
        unpack_packet(packet, x);

        const int valid_samples =
            valid_samples_for_packet(packet_idx);

        packet_sum_t packet_sum =
            (packet_sum_t)sum_abs16(x, 0, valid_samples);

        if (SAMPLES_PER_PACKET > 16) {
            packet_sum +=
                (packet_sum_t)sum_abs16(x, 16, valid_samples);
        }

        if (SAMPLES_PER_PACKET > 32) {
            packet_sum +=
                (packet_sum_t)sum_abs16(x, 32, valid_samples);
        }

        if (SAMPLES_PER_PACKET > 48) {
            packet_sum +=
                (packet_sum_t)sum_abs16(x, 48, valid_samples);
        }

        mav_sum += (mav_sum_t)packet_sum;

        // Forward the same physical packet to the Peak stage.
        output_stream.write(packet);

        if (packet_idx == MODEL_PACKETS_PER_WINDOW - 1) {
            mav_result_stream.write(
                mav_from_sum(mav_sum, mav_shift)
            );

            mav_sum = 0;
            packet_idx = 0;
        }
        else {
            packet_idx++;
        }
    }
}


// ============================================================
// STAGE 3
// FIFO -> Peak task -> FIFO
// ============================================================

static void peak_task(
    hls::stream<sample_packet_t> &input_stream,
    hls::stream<sample_packet_t> &output_stream,
    hls::stream<feature_t> &peak_result_stream,
    int total_packets
) {
#pragma HLS INLINE off

    const raw_t RAW_MIN =
        (raw_t)(-(1 << (SAMPLE_INT_BITS - 1)));

    raw_t running_peak = RAW_MIN;
    int packet_idx = 0;

PEAK_PACKET_LOOP:
    for (int idx = 0; idx < total_packets; idx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=1228800

        sample_packet_t packet = input_stream.read();

        raw_t x[MAX_SAMPLES_PER_PACKET];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
        unpack_packet(packet, x);

        const int valid_samples =
            valid_samples_for_packet(packet_idx);

        raw_t packet_peak = peak16(x, 0, valid_samples);

        if (SAMPLES_PER_PACKET > 16) {
            raw_t p = peak16(x, 16, valid_samples);
            if (p > packet_peak)
                packet_peak = p;
        }

        if (SAMPLES_PER_PACKET > 32) {
            raw_t p = peak16(x, 32, valid_samples);
            if (p > packet_peak)
                packet_peak = p;
        }

        if (SAMPLES_PER_PACKET > 48) {
            raw_t p = peak16(x, 48, valid_samples);
            if (p > packet_peak)
                packet_peak = p;
        }

        if (packet_peak > running_peak)
            running_peak = packet_peak;

        // Forward the same physical packet to the ZC stage.
        output_stream.write(packet);

        if (packet_idx == MODEL_PACKETS_PER_WINDOW - 1) {
            peak_result_stream.write((feature_t)running_peak);

            running_peak = RAW_MIN;
            packet_idx = 0;
        }
        else {
            packet_idx++;
        }
    }
}


// ============================================================
// STAGE 4
// FIFO -> Zero-Crossing task
// ============================================================
//
// Padded samples in a partial final packet are never included. This is
// especially important for 12-bit packing, where many window sizes are not
// exact multiples of 20 samples/logical packet.
static void zero_crossing_task(
    hls::stream<sample_packet_t> &input_stream,
    hls::stream<feature_t> &zc_result_stream,
    int total_packets
) {
#pragma HLS INLINE off

    ap_uint<18> zc_count = 0;
    raw_t prev = 0;
    bool have_prev = false;
    int packet_idx = 0;

ZC_PACKET_LOOP:
    for (int idx = 0; idx < total_packets; idx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=1228800

        sample_packet_t packet = input_stream.read();

        raw_t x[MAX_SAMPLES_PER_PACKET];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
        unpack_packet(packet, x);

        const int valid_samples =
            valid_samples_for_packet(packet_idx);

        ap_uint<7> packet_crossings =
            (ap_uint<7>)zc16(x, 0, valid_samples, prev, have_prev);

        if (SAMPLES_PER_PACKET > 16) {
            packet_crossings +=
                (ap_uint<7>)zc16(
                    x, 16, valid_samples,
                    x[15], valid_samples > 16
                );
        }

        if (SAMPLES_PER_PACKET > 32) {
            packet_crossings +=
                (ap_uint<7>)zc16(
                    x, 32, valid_samples,
                    x[31], valid_samples > 32
                );
        }

        if (SAMPLES_PER_PACKET > 48) {
            packet_crossings +=
                (ap_uint<7>)zc16(
                    x, 48, valid_samples,
                    x[47], valid_samples > 48
                );
        }

        zc_count += packet_crossings;

        if (valid_samples > 0) {
            prev = x[valid_samples - 1];
            have_prev = true;
        }

        if (packet_idx == MODEL_PACKETS_PER_WINDOW - 1) {
            zc_result_stream.write((feature_t)zc_count);

            zc_count = 0;
            prev = 0;
            have_prev = false;
            packet_idx = 0;
        }
        else {
            packet_idx++;
        }
    }
}


// ============================================================
// STAGE 5
// Join the three feature-result streams by window
// ============================================================

static void join_features(
    hls::stream<feature_t> &mav_result_stream,
    hls::stream<feature_t> &peak_result_stream,
    hls::stream<feature_t> &zc_result_stream,
    hls::stream<feature_packet_t> &feature_stream,
    int window_count
) {
#pragma HLS INLINE off

JOIN_FEATURE_LOOP:
    for (int window = 0; window < window_count; window++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=9600

        feature_packet_t feat;

        feat.mav  = mav_result_stream.read();
        feat.peak = peak_result_stream.read();
        feat.zc   = zc_result_stream.read();

        feature_stream.write(feat);
    }
}


// ============================================================
// STAGE 6
// Features -> 10-class MLR -> prediction stream
// ============================================================

static void classify_windows(
    hls::stream<feature_packet_t> &feature_stream,
    hls::stream<int> &prediction_stream,
    int window_count
) {
#pragma HLS INLINE off

CLASSIFY_WINDOW_LOOP:
    for (int window = 0; window < window_count; window++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=9600

        feature_packet_t packet =
            feature_stream.read();

        feature_t feat[NUM_FEATURES];
#pragma HLS ARRAY_PARTITION variable=feat complete dim=1

        feat[0] = packet.peak;
        feat[1] = packet.mav;
        feat[2] = packet.zc;

        acc_t score[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=score complete dim=1

INIT_SCORE_LOOP:
        for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS PIPELINE II=1
            score[c] = (acc_t)LR_HW_BIAS[c];
        }

        // Each feature column has its own 18-bit coefficient Q-format.
        // This preserves fractional precision for small ZC coefficients even
        // when Peak/MAV require many integer bits after scaler folding.
PEAK_MAC_LOOP:
        for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS PIPELINE II=1
            score[c] +=
                (acc_t)(packet.peak * LR_PEAK_WEIGHTS[c]);
        }

MAV_MAC_LOOP:
        for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS PIPELINE II=1
            score[c] +=
                (acc_t)(packet.mav * LR_MAV_WEIGHTS[c]);
        }

ZC_MAC_LOOP:
        for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS PIPELINE II=1
            score[c] +=
                (acc_t)(packet.zc * LR_ZC_WEIGHTS[c]);
        }

        int best_class = 0;
        acc_t best_score = score[0];

ARGMAX_LOOP:
        for (int c = 1; c < NUM_CLASSES; c++) {
#pragma HLS PIPELINE II=1

            if (score[c] > best_score) {
                best_score = score[c];
                best_class = c;
            }
        }

        prediction_stream.write(best_class);
    }
}


// ============================================================
// STAGE 7
// Prediction stream -> DDR
// ============================================================

static void write_predictions(
    hls::stream<int> &prediction_stream,
    int *predictions,
    int window_count
) {
#pragma HLS INLINE off

WRITE_PREDICTION_LOOP:
    for (int window = 0; window < window_count; window++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=9600

        int prediction = prediction_stream.read();
        predictions[window] = prediction;
    }
}


// Each feature task also emits one feature result per window.
// The three result streams are joined before MLR classification.
static void run_dataflow_pipeline(
    hls::stream<axis_bus_word_t> &input_stream0,
    hls::stream<axis_bus_word_t> &input_stream1,
    hls::stream<axis_bus_word_t> &input_stream2,
    hls::stream<axis_bus_word_t> &input_stream3,
    int *predictions,
    int window_count,
    int total_packets,
    int mav_shift
) {
#pragma HLS INLINE off

    hls::stream<sample_packet_t> packet_to_mav;
    hls::stream<sample_packet_t> packet_mav_to_peak;
    hls::stream<sample_packet_t> packet_peak_to_zc;

    hls::stream<feature_t> mav_result_stream;
    hls::stream<feature_t> peak_result_stream;
    hls::stream<feature_t> zc_result_stream;

    hls::stream<feature_packet_t> feature_stream;
    hls::stream<int> prediction_stream;

#pragma HLS STREAM variable=packet_to_mav depth=8
#pragma HLS STREAM variable=packet_mav_to_peak depth=8
#pragma HLS STREAM variable=packet_peak_to_zc depth=8

#pragma HLS STREAM variable=mav_result_stream depth=4
#pragma HLS STREAM variable=peak_result_stream depth=4
#pragma HLS STREAM variable=zc_result_stream depth=4

#pragma HLS STREAM variable=feature_stream depth=4
#pragma HLS STREAM variable=prediction_stream depth=4

#pragma HLS DATAFLOW

    read_four_streams(
        input_stream0,
        input_stream1,
        input_stream2,
        input_stream3,
        packet_to_mav,
        total_packets
    );

    mav_task(
        packet_to_mav,
        packet_mav_to_peak,
        mav_result_stream,
        total_packets,
        mav_shift
    );

    peak_task(
        packet_mav_to_peak,
        packet_peak_to_zc,
        peak_result_stream,
        total_packets
    );

    zero_crossing_task(
        packet_peak_to_zc,
        zc_result_stream,
        total_packets
    );

    join_features(
        mav_result_stream,
        peak_result_stream,
        zc_result_stream,
        feature_stream,
        window_count
    );

    classify_windows(
        feature_stream,
        prediction_stream,
        window_count
    );

    write_predictions(
        prediction_stream,
        predictions,
        window_count
    );
}


// ============================================================
// TOP
// ============================================================

void logistic_fault_predict(
    hls::stream<axis_bus_word_t> &input_stream0,
    hls::stream<axis_bus_word_t> &input_stream1,
    hls::stream<axis_bus_word_t> &input_stream2,
    hls::stream<axis_bus_word_t> &input_stream3,
    int *predictions,
    int window_count,
    int window_size
) {

#pragma HLS INTERFACE axis port=input_stream0
#pragma HLS INTERFACE axis port=input_stream1
#pragma HLS INTERFACE axis port=input_stream2
#pragma HLS INTERFACE axis port=input_stream3

#pragma HLS INTERFACE \
    m_axi port=predictions \
    offset=slave \
    bundle=gmem1 \
    depth=9600 \
    max_write_burst_length=256 \
    num_write_outstanding=16 \
    latency=64 \
    max_widen_bitwidth=64

#pragma HLS INTERFACE s_axilite port=predictions bundle=control
#pragma HLS INTERFACE s_axilite port=window_count bundle=control
#pragma HLS INTERFACE s_axilite port=window_size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (
        window_count <= 0 ||
        window_count > MAX_WINDOWS ||
        !supported_window_size(window_size) ||
        window_size != MODEL_WINDOW_SIZE
    ) {
        return;
    }

    const int mav_shift =
        window_shift(MODEL_WINDOW_SIZE);

    const int total_packets =
        window_count * MODEL_PACKETS_PER_WINDOW;

    run_dataflow_pipeline(
        input_stream0,
        input_stream1,
        input_stream2,
        input_stream3,
        predictions,
        window_count,
        total_packets,
        mav_shift
    );
}
