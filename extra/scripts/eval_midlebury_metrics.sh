GT_PATH=""
CALC_DISP=""
SAMPLES=($GT_PATH/*)
RESULT_DIR="midd_eval"

mkdir -p $RESULT_DIR
mkdir -p "$RESULT_DIR/gt"
mkdir -p "$RESULT_DIR/pred"

for SAMPLE in "${SAMPLES[@]}"; do
    SAMPLE_NAME=$(basename "$SAMPLE")
    ./pfm2png "$SAMPLE/disp0.pfm" "${RESULT_DIR}/gt/${SAMPLE_NAME}.png"
    ./pfm2png "$CALC_DISP/$SAMPLE_NAME.pfm" "${RESULT_DIR}/pred/${SAMPLE_NAME}.png"
done

for BAD_THS in $(LANG=en_US seq 0 0.5 5); do
    printf "%-10s $(./evaldisp $BAD_THS)\n" "sample" > "${RESULT_DIR}/${BAD_THS}.csv"
    for SAMPLE in "${SAMPLES[@]}"; do
        SAMPLE_NAME=$(basename "$SAMPLE")
        printf "%-10s $(./evaldisp "$CALC_DISP/$SAMPLE_NAME.pfm" "$SAMPLE/disp0.pfm" "$BAD_THS")\n" $SAMPLE_NAME >> "${RESULT_DIR}/${BAD_THS}.csv"
    done
done
