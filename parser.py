

if __name__ == "__main__":
    import os

    with open(os.path.join(os.getcwd(),"README.md"), "r") as f:
        lines = f.readlines()

    bullets = [line for line in lines if line.startswith("-")]
    
    for line in bullets:
        print(line.split("-")[1],)