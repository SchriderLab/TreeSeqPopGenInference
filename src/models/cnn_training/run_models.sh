for data in 128 256 512
do
    for model in 2dcnn 1dcnn
    do
        for nb in 1 2 3 4 5
        do 
            python demo_cnn_keras.py --in_train /pine/scr/d/d/ddray/demo_n${data}_1c.hdf5 --in_val /pine/scr/d/d/ddray/demo_n${data}_1c_val.hdf5 --conv_blocks ${nb} --model ${model}
            python demo_cnn_keras.py --in_train /pine/scr/d/d/ddray/demo_n${data}_1c.hdf5 --in_val /pine/scr/d/d/ddray/demo_n${data}_1c_val.hdf5 --conv_blocks ${nb} --model ${model} --nolog
            
            python demo_cnn_keras.py --in_train /pine/scr/d/d/ddray/demo_n${data}_1c_cosine.hdf5 --in_val /pine/scr/d/d/ddray/demo_n${data}_1c_cosine_val.hdf5 --conv_blocks ${nb} --model ${model}
            python demo_cnn_keras.py --in_train /pine/scr/d/d/ddray/demo_n${data}_1c_cosine.hdf5 --in_val /pine/scr/d/d/ddray/demo_n${data}_1c_cosine_val.hdf5 --conv_blocks ${nb} --model ${model} --nolog
        
        done 
    done
done