import os
import subprocess

def run_hasara():
    
    hasara_dir = os.path.join(os.getcwd(), "New_Hasara_Lite")

    print("\nStarting Hasara Program...\n")

    subprocess.run(
        ["python3", "src/main_controller.py"],
        cwd=hasara_dir
    )


def main():

    while True:

        print("\n==============================")
        print(" HASARA ENERGY MANAGEMENT ")
        print("==============================")
        print("1. Run Hasara Program")
        print("2. Exit")
        print("==============================")

        option = input("Enter option: ")

        if option == "1":
            run_hasara()

        elif option == "2":
            print("Exiting...")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    main()