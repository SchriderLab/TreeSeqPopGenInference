for conv in 1d 2d
do
    for pad in 128_cosine 256_cosine
    do
        sbatch --partition=dschridelab --gres=gpu:1 --time=02:00:00 --mem=64G -J ${pad}_${conv} -o logfiles/rho_${pad}_${conv}.out -e logfiles/rho_${pad}_${conv}.err \
            --wrap="python rho_cnn_keras.py \
            --in-train four_problems/recombination/n${pad}.hdf5 \
            --in-val four_problems/recombination/n${pad}_val.hdf5 \
            --net ${conv}"
    done
done