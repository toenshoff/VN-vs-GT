### generate datasets
python generate_data_cor49.py

###just make sure this arrays are same with the ones used for generation
test_l=(1 2 5 10 15 20 25 30 35 40 45 50)
test_r=(1 2 5 10 15 20 25 30 35 40 45 50)

models=(VN GPS)

for model in "${models[@]}"; do
  python main.py --cfg configs/Cor49-${model}/train.yaml
  ### run all test data
  for l in "${test_l[@]}"; do
    for r in "${test_r[@]}"; do
      python custom_test_cor49.py --cfg ./configs/Cor49-${model}/test.yaml wandb.use False dataset.format PyG-Cor49-${l}-${r}
    done
  done
done
