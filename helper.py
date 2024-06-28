import glob
import os

# Get a list of all .py files in the current directory and subdirectories
py_files = glob.glob('**/*.py', recursive=True)

# Open the output file
with open('output.txt', 'w') as output_file:
    # Iterate over the list of .py files
    for py_file in py_files:
        # Write the file path to the output file
        output_file.write(f'# {os.path.abspath(py_file)}\n')
        
        # Open the .py file and write its contents to the output file
        with open(py_file, 'r') as input_file:
            output_file.write(input_file.read())
            output_file.write('\n\n')