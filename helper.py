import os

def combine_files(directory):
    combined_content = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                combined_content += f"# {file_path}\n{content}\n\n"
    return combined_content

# Use the function
directory = '.'  # current directory
combined_content = combine_files(directory)

# Write the combined content to a new file
with open('combined.py', 'w') as f:
    f.write(combined_content)