# Read the contents of both the pip freeze output and requirements.txt
with open('current_freeze.txt', 'r', encoding='utf-16') as freeze_file:
    freeze_packages = set(freeze_file.read().splitlines())

with open('requirements.txt', 'r', encoding='utf-16') as req_file:
    required_packages = set(req_file.read().splitlines())

# Find the difference between the two sets (those in freeze but not in requirements)
difference = freeze_packages - required_packages
print(difference)

# Write the difference to a new text file
with open('revert_venv.txt', 'w') as diff_file:
    for package in difference:
        diff_file.write(package + '\n')

print("Comparison complete.")