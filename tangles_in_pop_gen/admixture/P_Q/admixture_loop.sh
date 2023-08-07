#!/bin/bash

file_name="$1"
K="$2"
vcf_path="/home/klara/ML_in_pop_gen_in_process/tangles_in_pop_gen/tangles_in_pop_gen/admixture/data/${file_name}"
echo "${vcf_path}"
vcf_file="${vcf_path}.vcf"
echo "$vcf_file"
bed_file="${vcf_path}.bed"
echo "$bed_file"

plink --vcf "$vcf_file" --make-bed --out "$vcf_path"

echo 'start admixture.'
output_dir="/home/klara/ML_in_pop_gen_in_process/tangles_in_pop_gen/tangles_in_pop_gen/admixture/P_Q/"
admixture_path="/home/klara/ML_in_pop_gen_in_process/tangles_in_pop_gen/tangles_in_pop_gen/admixture/admixture"
mkdir -p "$output_dir"
cd "$output_dir"
"$admixture_path" "$bed_file" "$K"
echo 'admixture done.'



