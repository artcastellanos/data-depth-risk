from datetime import date, datetime
import joblib

def get_timestamp():
    now = datetime.now()
    today = date.today()
    current_time = now.strftime("%Hh%Mm%S")
    day = today.strftime("%d-%m-%Y")
    timestamp = day + '_' + current_time
    return timestamp


def saving_path(name,prefix_folder='save/',timebool=True):
    timestamp = ''
    if timebool:
        timestamp = get_timestamp()

    save_path = prefix_folder + timestamp + name
    return save_path


def saving(object,name,prefix_folder='save/',timebool=False):
    save_path = saving_path(name,prefix_folder,timebool)
    joblib.dump(object, save_path)

def loader(file_name,prefix_folder='save/'):
    save_path = prefix_folder + file_name
    results = joblib.load(save_path)
    return results


if __name__ == '__main__':
    l = [1,2,3]
    saving(l,'test123')