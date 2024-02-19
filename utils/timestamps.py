from datetime import datetime


def get_stamp():
    now = str(datetime.now())
    stamp = now.replace(':','-').replace('.', '-').replace(' ', '-')
    return stamp