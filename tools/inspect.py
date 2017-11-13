import sys
sys.path.append(sys.path[0])
del sys.path[0]
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name="/media/bnrc2/_backup/golf/resume/tiny_hourglass_19", tensor_name='', all_tensors=False)