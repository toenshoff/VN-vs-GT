### generate datasets
python generate_data_cor44.py

###just make sure this arrays are same with the ones used for generation
test_num_nodes=(10 20 30 40 50 60 70 80 90 100 120 140 160 180 200)
models=(VN GPS)

for model in "${models[@]}"; do
  python main.py --cfg configs/Cor44-${model}/train.yaml
  ### run all test data
  for n in "${test_num_nodes[@]}"; do
      python custom_test_cor44.py --cfg ./configs/Cor44-${model}/test.yaml wandb.use False dataset.format PyG-Cor44-${n}
  done
done
