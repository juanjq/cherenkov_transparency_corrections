#! /bin/bash

file="/fefs/aswg/workspace/juan.jimenez/cosmic_ray_data_correction/bash_weather_data/joblist.txt"

# read the lines of the given file
while read -r line; do

# only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
str="$line"

echo "#! /bin/bash" > dl2_index_ws_job_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/cosmic_ray_data_correction/bash_weather_data/script_generate_indexes.py '$str'" >> dl2_index_ws_job_tmpjob.sh

echo -e "Sending job $str to convert dl1 to dl2 with different light scaling"
sbatch -p short --output="/fefs/aswg/workspace/juan.jimenez/cosmic_ray_data_correction/bash_weather_data/output_slurm/slurm-%j.out" dl2_index_ws_job_tmpjob.sh

fi

done < $file
