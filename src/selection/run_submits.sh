while read dataset_idx
do
    sbatch submit_seriates.sh $dataset_idx
done < dataset_labels.txt