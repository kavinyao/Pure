# Levels
ERROR = 4
CRITICAL = 3
WARNING = 2
INFO = 1

current_level = INFO

def log(msg, level=INFO):
    if level >= current_level:
        print msg
