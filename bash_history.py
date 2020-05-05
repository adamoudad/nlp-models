from pathlib import Path
history_path = Path.home() / ".bash_history"



def load_data(path=history_path):
    return [ line.rstrip("\n") for line in path.open() ]
