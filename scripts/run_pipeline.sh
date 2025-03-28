#! /bin/bash
DATASET_NAME="v1.0-mini"
MODEL="neuflow"
SAMPLE_ID=100
while getopts ':ds:m:i:' OPTION; do
  case "$OPTION" in
    s)
      RUN_SPLITDATA=1
      ;;
    d)
      DATASET_NAME="$OPTARG"
      ;;
    m)
      MODEL="$OPTARG"
      ;;
    i)
      SAMPLE_ID="$OPTARG"
      ;;
    ?)
      echo "script usage: $(basename \$0) [-s (run split-data)] [-d dataset_name (provide name of dataset)] [-m neuflow|raft|skip] [-i sample_id (id to display)]" >&2
      exit 1
      ;;
  esac
done

if [ $RUN_SPLITDATA ]; then
    python3 split_data.py --version $DATASET_NAME
    python3 prepare_flow_im.py --version $DATASET_NAME
fi

if [[ "$MODEL" != "neuflow" && "$MODEL" != "raft" && "$MODEL" != "skip" ]]; then
    echo $MODEL
    echo "Error: Invalid model. Accepted values are 'neuflow' or 'raft' or 'skip'."
    usage
    exit
fi


if [ $MODEL == "raft" ]; then
    python3 cal_flow.py
elif [ $MODEL == "neuflow" ]; then
    python3 cal_flow_neuflow.py
fi

if [ $MODEL != "skip" ]; then
    python3 cal_im_flow2uv.py --version $DATASET_NAME
    python3 gen_h5_file.py
    python3 test_association.py  --gen_offset --version $DATASET_NAME
fi
python3 prd_full_v.py --version $DATASET_NAME --sample_idx $SAMPLE_ID
