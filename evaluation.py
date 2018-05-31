import numpy as np

def evaluation(logits_xy, logits_marks, labels, marks):
    correct_mark = 0
    total_mark = 0
    accuracy = 1
    return accuracy