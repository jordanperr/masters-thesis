python 1_generate_experiment_list.py $1
python 2_generate_student_networks.py $1
python 3_verify_student_networks.py $1
python 4_collect_data $1
python 5_verify_teacher_network $1