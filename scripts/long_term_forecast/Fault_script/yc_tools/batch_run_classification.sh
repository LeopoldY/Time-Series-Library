seq_len_list=(12 24 48 96 168)

for seq_len in "${seq_len_list[@]}"
do
    echo "Processing squence length: $seq_len"
    python -u scripts/long_term_forecast/Fault_script/yc_tools/run_classification.py \
      --squence_length $seq_len 
done
