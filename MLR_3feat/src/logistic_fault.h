#ifndef LOGISTIC_FAULT_H
#define LOGISTIC_FAULT_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// ------------------------------------------------------------
// Compile-time model selector.
// CHANGE ONLY THIS #define when switching sweep configurations.
// The master logistic_params.h contains every trained model block.
// ------------------------------------------------------------
// Available macro pattern: MODEL_W<window>_B<bits>
#define MODEL_W512_B4

#include "logistic_params.h"

#define NUM_INPUT_BUSES 4
#define BUS_BITS 64

// Max-pack every 64-bit input bus at compile time.
//   SAMPLE_BITS=4  -> 16 samples/bus -> 64 samples/logical packet
//   SAMPLE_BITS=8  ->  8 samples/bus -> 32 samples/logical packet
//   SAMPLE_BITS=12 ->  5 samples/bus -> 20 samples/logical packet
//   SAMPLE_BITS=16 ->  4 samples/bus -> 16 samples/logical packet
#define SAMPLES_PER_BUS (BUS_BITS / SAMPLE_BITS)
#define SAMPLES_PER_PACKET (NUM_INPUT_BUSES * SAMPLES_PER_BUS)
#define MAX_SAMPLES_PER_PACKET 64

typedef ap_uint<BUS_BITS> bus_word_t;
typedef ap_axiu<BUS_BITS, 0, 0, 0> axis_bus_word_t;
typedef ap_uint<NUM_INPUT_BUSES * BUS_BITS> sample_packet_t;

// SAMPLE_INT_BITS=4 is held constant across the sweep, so the formats are
// Q4.0/Q4.4/Q4.8/Q4.12 and increasing SAMPLE_BITS adds fractional resolution.
typedef ap_fixed<SAMPLE_BITS,SAMPLE_INT_BITS,AP_RND,AP_SAT> raw_t;

// One extra integer/total bit safely holds b-a and |min(raw_t)|.
typedef ap_fixed<SAMPLE_BITS + 1,SAMPLE_INT_BITS + 1,AP_RND,AP_SAT> internal_t;

typedef ap_fixed<SAMPLE_BITS + 1,SAMPLE_INT_BITS + 1,AP_RND,AP_SAT> magnitude_t;

// Wider accumulator for the 10-class MLR scores.
typedef ap_fixed<48, 32, AP_RND, AP_SAT> acc_t;

// MAV accumulation preserves the raw sample binary-point position.
// Starting from magnitude_t, summing as many as 2048 (=2^11) samples
// requires 11 more integer/total bits.
#define MAV_SUM_TOTAL_BITS (SAMPLE_BITS + 12)
#define MAV_SUM_INT_BITS   (SAMPLE_INT_BITS + 12)
typedef ap_fixed<MAV_SUM_TOTAL_BITS,MAV_SUM_INT_BITS,AP_RND,AP_SAT> mav_sum_t;

struct feature_packet_t {
    feature_t peak;
    feature_t mav;
    feature_t zc;
};

#define MAX_WINDOWS 9600
#define MAX_WINDOW_SIZE 2048

#define WINDOW_256 256
#define WINDOW_512 512
#define WINDOW_1024 1024
#define WINDOW_2048 2048

// Ceiling packet count. This matters for 12-bit packing because, for example,
// 1024 samples / 20 samples-per-packet requires 52 packets.
#define MODEL_PACKETS_PER_WINDOW \
    ((MODEL_WINDOW_SIZE + SAMPLES_PER_PACKET - 1) / SAMPLES_PER_PACKET)

// Number of real samples in the final packet
#define LAST_PACKET_VALID_SAMPLES \
    (MODEL_WINDOW_SIZE - ((MODEL_PACKETS_PER_WINDOW - 1) * SAMPLES_PER_PACKET))

#define MAX_PACKETS_PER_WINDOW \
    ((MAX_WINDOW_SIZE + SAMPLES_PER_PACKET - 1) / SAMPLES_PER_PACKET)

static_assert(
    SAMPLE_BITS == 4 || SAMPLE_BITS == 8 ||
    SAMPLE_BITS == 12 || SAMPLE_BITS == 16,
    "SAMPLE_BITS must be 4, 8, 12, or 16"
);

static_assert(
    SAMPLE_INT_BITS == 4,
    "This unscaled fixed-point sweep expects SAMPLE_INT_BITS=4"
);

static_assert(
    SAMPLE_FRAC_BITS == SAMPLE_BITS - SAMPLE_INT_BITS,
    "SAMPLE_FRAC_BITS must match SAMPLE_BITS-SAMPLE_INT_BITS"
);

static_assert(
    SAMPLES_PER_BUS * SAMPLE_BITS <= BUS_BITS,
    "Packed samples must fit in each 64-bit AXI stream word"
);

static_assert(
    MODEL_WINDOW_SIZE == WINDOW_256 ||
    MODEL_WINDOW_SIZE == WINDOW_512 ||
    MODEL_WINDOW_SIZE == WINDOW_1024 ||
    MODEL_WINDOW_SIZE == WINDOW_2048,
    "MODEL_WINDOW_SIZE must be 256, 512, 1024, or 2048"
);

void logistic_fault_predict(
    hls::stream<axis_bus_word_t> &input_stream0,
    hls::stream<axis_bus_word_t> &input_stream1,
    hls::stream<axis_bus_word_t> &input_stream2,
    hls::stream<axis_bus_word_t> &input_stream3,
    int *predictions,
    int window_count,
    int window_size
);

#endif
