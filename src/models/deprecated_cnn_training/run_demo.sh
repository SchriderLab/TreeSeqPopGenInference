for encoding in 01 neg11 0255
do
    for data in 128 256
    do
        for model in 2d 1d
        do
            for nb in 1 2 3
            do 
                sbatch --partition=dschridelab --gres=gpu:1 --time=02:00:00 --mem=32G -J ${data}_${model}_${nb} -o logfiles/${data}_${model}_${nb}.out -e logfiles/${data}_${model}_${nb}.err \
                    --wrap="python demo_cnn_keras.py \
                    --in-train four_problems/demography/n${data}_cosine.hdf5 \
                    --in-val four_problems/demography/n${data}_cosine_val.hdf5 \
                    --conv-blocks ${nb} \
                    --net ${model} \
                    --encoding ${encoding}"
                
            done 
        done
    done
done