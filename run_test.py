import subprocess
import argparse


def run_test():
    parser = argparse.ArgumentParser(description="Choose which test script to run.")
    parser.add_argument("test_script", help="name of the test function to run.")
    args = parser.parse_args()

    test_script = args.test_script
    test_script_path = 'tests/' + test_script

    try:
        subprocess.run(['python', test_script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    else:
        print(f"Script {test_script} executed successfully")


run_test()
