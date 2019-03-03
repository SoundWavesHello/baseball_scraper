# from flask import Flask
from pybaseball import statcast
import pandas as pd
import csv

# print a nice greeting.
def getInfo():
    print("In GET INFO")

    data = statcast(start_dt='2018-03-29', end_dt='2018-09-30')
    # print(data.head(data.size))
    data.to_csv('total_2018.csv')


def csvPreProcessing():

	header_dict = {'description': 1,'hit_location': 1,'on_3b': 1,'on_2b': 1,'on_1b': 1,'outs_when_up': 1,'hc_x': 1,'hc_y': 1,'hit_distance': 1,'launch_speed': 1,'launch_angle': 1,'fielder_3': 1,'fielder_4': 1,'fielder_5': 1,'fielder_6': 1,'fielder_7': 1,'fielder_8': 1,'fielder_9': 1,'if_fielding_alignment': 1,'of_fielding_alignment': 1,'bat_score': 1,'post_bat_score': 1,'game_date': 1, 'home_team':1, 'batter':1, 'estimated_ba_using_speedangle':1, 'inning_topbot':1}
	with open('total_2018.csv') as file_r:
		with open('data_2018.csv','a') as file_w:
			# writer = csv.writer(file_w)

			#file_w.write(myCsvRow)
			reader = csv.DictReader(file_r)
			writer = csv.DictWriter(file_w, header_dict.keys())
			writer.writeheader()
			hit_flag = 0
			for old_row in reversed(list(reader)):
				attributes = ['description','hit_location','on_3b','on_2b','on_1b','outs_when_up','hc_x','hc_y','hit_distance','launch_speed','launch_angle','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7','fielder_8','fielder_9','if_fielding_alignment','of_fielding_alignment','bat_score','post_bat_score','game_date', 'home_team', 'batter', 'estimated_ba_using_speedangle', 'inning_topbot']
				if old_row['description'] == 'hit_into_play' or old_row['description'] == 'hit_into_play_no_out': # store the hit data
					hit_flag = 2
				if hit_flag > 0:
					new_row = {}
					for attribute, data in old_row.items():
						if attribute in attributes:
							new_row[attribute] = data
					writer.writerow(new_row)
					hit_flag -= 1
	file_r.close()
	file_w.close()


				# for attribute, data in old_row.items():
				# 	new_row = {}
				# 	if attribute == 'description' and (data == 'hit_into_play' or data == 'hit_into_play_no_out'): 
				# 		hit_flag = True
				# 	if attribute in attributes:
				# 		print('HERE')
				# 		new_row[attribute] = data
				# 		# print(new_row)
				# 		writer.writerow(new_row)
				# 	elif hit_flag == True: # previous play was a hit, store data for positions of runners
				# 		hit_flag = False
				# 		if attribute in attributes:
				# 			new_row[attribute] = data
				# 		writer.writerow(new_row)


# hit_location
# on_3b
# on_2b
# on_1b
# outs_when_up
# hc_x
# hc_y
# hit_distance
# launch_speed
# launch_angle
# fielder_3
# fielder_4
# fielder_5
# fielder_6
# fielder_7
# fielder_8
# fielder_9
# if_fielding_alignment
# of_fielding_alignment
# bat_score
# post_bat_score
# game_date


getInfo()
csvPreProcessing()

# EB looks for an 'application' callable by default.
# application = Flask(__name__)

# run the app.
# if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    # application.run()