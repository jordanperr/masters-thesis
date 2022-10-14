INPUT_DIR=../data
OUTPUT_DIR=output

mkdir output

python -m experiment.generate_student_networks $INPUT_DIR $OUTPUT_DIR
python -m experiment.verify_student_networks $INPUT_DIR $OUTPUT_DIR
