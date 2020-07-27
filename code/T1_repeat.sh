#!/usr/bin/env bash
echo "run T1_BT_Icub_joint_ae.py for multiple times"
echo "remember to check model name and dict_name in python script"

for i in {1..10}
do
#    python test.py -i $i
   python T1_BT19_Icub_joint_ae.py -k 0 -c 0 --data_dir /home/ruihan/data -i $i
done

echo "check multiple results"
echo "change the base_name and target csv_name in check_multiple_results.py"
python check_multiple_results.py