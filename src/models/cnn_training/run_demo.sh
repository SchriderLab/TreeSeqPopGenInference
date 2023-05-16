for data in 128 256
do
    for model in 2dcnn 1dcnn
    do
        for nb in 1 2 3
        do 
            python demo_cnn_keras.py \
                --in_train four_problems/demography/n${data}_cosine.hdf5 \
                --in_val four_problems/demography/n${data}_cosine_val.hdf5 \
                --conv_blocks ${nb} \
                --model ${model}
            
        done 
    done
done