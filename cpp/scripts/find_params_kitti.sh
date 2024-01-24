#!/bin/bash

PATH_TO_KITTI_DATASET=""
MATCHER_EXECUTABLE=""
OUTPUT_DIR="find_params_kitti"
START_GAPS=("-0.002" "-0.004" "-0.015" "-0.1")
CONTINUE_GAPS=("-0.008" "-0.005" "-0.02")
MATCHES=("0.012" "0.02" "0.1" "0.2" "0.05")
EDGES_THSS=("0.0001")
MAX_DISP="2000"
THREADS="8"
MULT="16"

mkdir -p "$OUTPUT_DIR"

SAMPLES=($(ls "${PATH_TO_KITTI_DATASET}"/image_0/*_10.png))

SAMPLE=${SAMPLES[0]}
SAMPLE=$(basename "$SAMPLE")

for START_GAP in "${START_GAPS[@]}"; do
    for CONTINUE_GAP in "${CONTINUE_GAPS[@]}"; do
        for MATCH in "${MATCHES[@]}"; do
            for EDGES_THS in "${EDGES_THSS[@]}"; do
                MATCH_TIME=$( (time \
                "$MATCHER_EXECUTABLE" \
                "${PATH_TO_KITTI_DATASET}"/image_0/${SAMPLE} \
                "${PATH_TO_KITTI_DATASET}"/image_1/${SAMPLE} \
                "${OUTPUT_DIR}/${SAMPLE}_${START_GAP}_${CONTINUE_GAP}_${MATCH}_${EDGES_THS}.png" \
                "$START_GAP" "$CONTINUE_GAP" "$MATCH" "$EDGES_THS" "$MAX_DISP" "$THREADS" "$MULT" \
                ) 2>&1 )

                echo "$SAMPLE" "$MATCH_TIME"
            done
        done
    done
done

