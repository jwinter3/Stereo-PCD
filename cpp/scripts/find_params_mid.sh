#!/bin/bash

PATH_TO_MIDDLEBURY_DATASET=""
MATCHER_EXECUTABLE=""
OUTPUT_DIR="find_params"
START_GAPS=("-0.002" "-0.004" "-0.015")
CONTINUE_GAPS=("-0.008" "-0.005")
MATCHES=("0.012" "0.02")
EDGES_THSS=("0.0001")
MAX_DISP="200"
THREADS="8"
MULT="16"

mkdir -p "$OUTPUT_DIR"

SAMPLES=($(ls "$PATH_TO_MIDDLEBURY_DATASET"))

SAMPLE=${SAMPLES[0]}

for START_GAP in "${START_GAPS[@]}"; do
    for CONTINUE_GAP in "${CONTINUE_GAPS[@]}"; do
        for MATCH in "${MATCHES[@]}"; do
            for EDGES_THS in "${EDGES_THSS[@]}"; do
            MATCH_TIME=$( (time \
            "$MATCHER_EXECUTABLE" \
            "${PATH_TO_MIDDLEBURY_DATASET}/${SAMPLE}/im0.png" \
            "${PATH_TO_MIDDLEBURY_DATASET}/${SAMPLE}/im1.png" \
            "${OUTPUT_DIR}/${SAMPLE}_${START_GAP}_${CONTINUE_GAP}_${MATCH}_${EDGES_THS}.png" \
            "$START_GAP" "$CONTINUE_GAP" "$MATCH" "$EDGES_THS" "$MAX_DISP" "$THREADS" "$MULT" \
            ) 2>&1 )

            echo "$SAMPLE" "$MATCH_TIME"
            done
        done
    done
done

