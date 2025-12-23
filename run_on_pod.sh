export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

SEEDS=(1557 831 121 2023 2024 2025 2026)

for index in "${!SEEDS[@]}"; do
    seed=${SEEDS[$index]}
    echo "Using seed = $seed in cuda device = $index"
    CUDA_VISIBLE_DEVICES="$index" python3 HyQuRP_light_modelnet,shapenet.py $seed > "modelnet_shapenet_$seed.dat" 2> modelnet_shapenet_$seed.err &
done
wait

cp modelnet_shapenet_*.dat /workspace
