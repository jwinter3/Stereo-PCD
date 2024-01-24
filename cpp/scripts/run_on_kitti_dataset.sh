#!/bin/bash

PATH_TO_KITTI_DATASET=""
MATCHER_EXECUTABLE=""
OUTPUT_DIR="out_kitti"

START_GAP="-0.015"
CONTINUE_GAP="-0.005"
MATCH="0.05"
EDGES_THS="0.0001"
MAX_DISP="2000"
THREADS="1"
MULT="16"

mkdir -p "$OUTPUT_DIR"

SAMPLES=($(ls "${PATH_TO_KITTI_DATASET}"/image_0/*_10.png))

for SAMPLE in ${SAMPLES[@]}; do
    SAMPLE=$(basename "$SAMPLE")

    MATCH_TIME=$( (time \
    "$MATCHER_EXECUTABLE" \
    "${PATH_TO_KITTI_DATASET}"/image_0/${SAMPLE} \
    "${PATH_TO_KITTI_DATASET}"/image_1/${SAMPLE} \
    "${OUTPUT_DIR}/${SAMPLE}" \
    "$START_GAP" "$CONTINUE_GAP" "$MATCH" "$EDGES_THS" "$MAX_DISP" "$THREADS"  "$MULT" \
    ) 2>&1 )

    echo "$SAMPLE" "$MATCH_TIME"

done

