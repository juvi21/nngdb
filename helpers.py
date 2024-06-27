import glob

# Use glob to find all .py files in current directory and subdirectories
py_files = glob.glob('**/*.py', recursive=True)

# Open output.txt file in write mode
with open('output.txt', 'w') as f:
    # Write each file path to output.txt
    for file in py_files:
        f.write('# ' + file + '\n')
        with open(file, 'r') as py_file:
            f.write(py_file.read() + '\n\n')