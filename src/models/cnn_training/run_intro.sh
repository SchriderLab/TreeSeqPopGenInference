for conv in 1d 2d
do
    for mt in mt2_4 mt3_5
    do
        for pad in 256_cosine 512_cosine
        do
            sbatch --partition=dschridelab --gres=gpu:1 --time=02:00:00 --mem=32G -J ${mt}_${pad}_${conv} -o logfiles/${mt}_${pad}_${conv}.out -e logfiles/${mt}_${pad}_${conv}.err \
                --wrap="python intro_cnn_keras.py \
                --in-train four_problems/dros/${mt}/n${pad}.hdf5 \
                --in-val four_problems/dros/${mt}/n${pad}_val.hdf5 \
                --out-prefix intro/${mt}_n${pad} \
                --net ${conv}"
        done
    done
done