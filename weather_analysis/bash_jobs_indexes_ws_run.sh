#! /bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/weather_analysis/objects/bash_job_list.txt"

# read the lines of the given file
while read -r line; do

# only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
str="$line"

echo "#! /bin/bash" > bash_indexes_ws_run_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/weather_analysis/script_relation_ws_run.py '$str'" >> bash_indexes_ws_run_tmpjob.sh

echo -e "Sending job $str to the queue...\n"
sbatch -p short --output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/weather_analysis/objects/output_slurm/slurm-%j.out" bash_indexes_ws_run_tmpjob.sh

fi

done < $file
