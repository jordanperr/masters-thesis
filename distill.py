# Helper functions and classes for distill_test example
import onnx
import onnx_tf
import tensorflow
import time

def load_network(path):
    """
    load network in onnx format
    """
    pass

def distill(teacher, student_options):
    """
    data-free distillation (compression) of teacher network
    """
    pass

def verify(network, property, verifier):
    """
    use an external verification tool to verify the network
    """
    pass

def distill_verify_comparison_experiment(network_path, property_path, student_options):
    teacher = load_network(network_path)
    
    # Classic Verification
    output = verify(student, property_path, timeout=600)

    # Distillation Verification
    student = distill(teacher, student_options)
    output = verify(teacher, property_path, timeout=600)

def refinement_loop_experiment():
    pass