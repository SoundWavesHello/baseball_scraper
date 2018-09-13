from flask import Flask

from pybaseball import statcast

# print a nice greeting.
def getInfo():
    print("In GET INFO")

    data = statcast(start_dt='2018-09-11', end_dt='2018-09-12')

    for item in data:
        print(item)


getInfo()

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()