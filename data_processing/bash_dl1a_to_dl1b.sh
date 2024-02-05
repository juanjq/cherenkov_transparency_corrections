#! /bin/bash

# The file with the list of jobs to be sent
file="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/config/job_config_runs.txt"

# read the lines of the given file
while read -r line; do

# only operating if the line is not commented with #
if  [[ "${line:0:1}" != '#' ]]; then
str="$line"

echo "#! /bin/bash" > bash_dl1a_to_dl1b_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/dl1a_to_dl1b_scaling.py '$str'" >> bash_dl1a_to_dl1b_tmpjob.sh
echo "python /fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/dl1a_to_dl1b.py '$str'" >> bash_dl1a_to_dl1b_tmpjob.sh

echo -e "Sending job $str to the queue...\n"
sbatch -p short --output="/fefs/aswg/workspace/juan.jimenez/cherenkov_transparency_corrections/data_processing/objects/output_slurm/slurm-%j.out" bash_dl1a_to_dl1b_tmpjob.sh

fi

done < $file