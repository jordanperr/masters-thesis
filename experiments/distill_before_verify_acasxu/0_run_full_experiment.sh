python 1_generate_experiment_list.py $1
python 2_generate_student_networks.py $1
python 3_verify_student_networks.py $1
python 4_verify_teacher_network.py $1
python 5_collect_data.py $1
python 6_collect_teacher_data.py $1
