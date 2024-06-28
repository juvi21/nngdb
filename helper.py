import os

output_file_path = 'output.txt'

# Log the current working directory
print(f'Running from: {os.getcwd()}')

with open(output_file_path, 'w') as output_file:
    for root, dirs, files in os.walk('.'):  # start from current directory
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                output_file.write(f'# {file_path}\n')
                with open(file_path, 'r') as input_file:
                    output_file.write(input_file.read())
                    output_file.write('\n')