# print the output of which python is being used
echo "Using python from $(which python)"

# Get the absolute path of the TOTEM directory
TOTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

seq_len=96
# Use absolute path for data directory
root_path_name="$TOTEM_DIR/../electricity"
data_path_name=electricity.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

# Check if data file exists
if [ ! -f "$root_path_name/$data_path_name" ]; then
    echo "Error: Data file not found at $root_path_name/$data_path_name"
    echo "Please ensure the electricity.csv file exists in the correct location"
    exit 1
fi

# Create necessary directories
mkdir -p "$TOTEM_DIR/forecasting/data/electricity"
mkdir -p "$TOTEM_DIR/forecasting/saved_models/electricity"
mkdir -p "$TOTEM_DIR/forecasting/results/electricity"

# First step: Save RevIN data
python -u "$TOTEM_DIR/forecasting/save_revin_data.py" \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "$root_path_name" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 321 \
  --gpu $gpu \
  --save_path "$TOTEM_DIR/forecasting/data/electricity"

# Second step: Train VQVAE
python "$TOTEM_DIR/forecasting/train_vqvae.py" \
  --config_path "$TOTEM_DIR/forecasting/scripts/electricity.json" \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_electricity \
  --save_path "$TOTEM_DIR/forecasting/saved_models/electricity/" \
  --base_path "$TOTEM_DIR/forecasting/data" \
  --batchsize 4096

# Store the path to the trained VQVAE model
VQVAE_MODEL_PATH="$TOTEM_DIR/forecasting/saved_models/electricity/vqvae_model.pt"

# Third step: Extract forecasting data
for pred_len in 96 192 336 720
do
    python -u "$TOTEM_DIR/forecasting/extract_forecasting_data.py" \
      --random_seed $random_seed \
      --data $data_name \
      --root_path "$root_path_name" \
      --data_path $data_path_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --label_len 0 \
      --enc_in 321 \
      --gpu $gpu \
      --save_path "$TOTEM_DIR/forecasting/data/electricity/Tin${seq_len}_Tout${pred_len}/" \
      --trained_vqvae_model_path "$VQVAE_MODEL_PATH" \
      --compression_factor 4 \
      --classifiy_or_forecast "forecast"
done

# Fourth step: Train forecaster
for seed in 2021 1 13
do
    for Tout in 96 192 336 720
    do
        python "$TOTEM_DIR/forecasting/train_forecaster.py" \
          --data-type electricity \
          --Tin $seq_len \
          --Tout $Tout \
          --cuda-id $gpu \
          --seed $seed \
          --data_path "$TOTEM_DIR/forecasting/data/electricity/Tin${seq_len}_Tout${Tout}" \
          --codebook_size 256 \
          --checkpoint \
          --checkpoint_path "$TOTEM_DIR/forecasting/saved_models/electricity/forecaster_checkpoints/electricity_Tin${seq_len}_Tout${Tout}_seed${seed}" \
          --file_save_path "$TOTEM_DIR/forecasting/results/electricity/"
    done
done