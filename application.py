# from flask import Flask
from pybaseball import statcast
import pandas as pd
import csv

# file names
all_data_file_name = 'raw_data_2018.csv'
processed_data_file_name = 'processed_data_2018.csv'

# print a nice greeting.
def getInfo(input_file):
    print("In GET INFO")

    data = statcast(start_dt='2018-03-28', end_dt='2018-10-1')
    # print(data.head(data.size))
    data.to_csv(input_file)



def csvPreProcessing(input_file, output_file):

	print("IN PRE-PROCESSING")

	header_dict = {'description': 1,'hit_location': 1,'on_3b': 1,'on_2b': 1,'on_1b': 1,'outs_when_up': 1,'hc_x': 1,'hc_y': 1,'hit_distance': 1,'launch_speed': 1,'launch_angle': 1,'fielder_3': 1,'fielder_4': 1,'fielder_5': 1,'fielder_6': 1,'fielder_7': 1,'fielder_8': 1,'fielder_9': 1,'if_fielding_alignment': 1,'of_fielding_alignment': 1,'bat_score': 1,'post_bat_score': 1,'game_date': 1, 'home_team':1, 'batter':1, 'estimated_ba_using_speedangle':1, 'inning_topbot':1, 'total_bases':1}
	with open(input_file) as file_r:
		with open(output_file,'a') as file_w:
			# writer = csv.writer(file_w)

			#file_w.write(myCsvRow)
			reader = csv.DictReader(file_r)
			writer = csv.DictWriter(file_w, header_dict.keys())
			writer.writeheader()

			revised = list(reader)
			revised.reverse()

			for i, old_row in enumerate(revised):
				if (i % 1000 == 0):
					print(i)
				attributes = ['description','hit_location','on_3b','on_2b','on_1b','outs_when_up','hc_x','hc_y','hit_distance_sc','launch_speed','launch_angle','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7','fielder_8','fielder_9','if_fielding_alignment','of_fielding_alignment','bat_score','post_bat_score','game_date', 'home_team', 'batter', 'estimated_ba_using_speedangle', 'inning_topbot']
				if old_row['description'] == 'hit_into_play' or old_row['description'] == 'hit_into_play_no_out': # store the hit data
					new_row = {}
					for attribute, data in old_row.items():
						if attribute in attributes:
							new_row[attribute] = data

					# get net bases
					next_event = revised[i+1]

					# if it's an inning ending play, then no bases were gained
					if old_row['inning_topbot'] != next_event['inning_topbot']:
						new_row['total_bases'] = 0
					else:
						# find difference in outs and scores
						net_outs = int(float(next_event['outs_when_up'])) - int(float(old_row['outs_when_up']))
						net_score = int(float(old_row['post_bat_score'])) - int(float(old_row['bat_score']))

						net_bases = 0

						# check 3B
						if (old_row['on_3b'] != '' and old_row['on_3b'] != next_event['on_3b']):
							if (net_score > 0):
								net_bases += 1
								net_score -= 1
							elif (net_outs > 0):
								net_bases -= 3
								net_outs -= 1

						# check 2B
						if (old_row['on_2b'] != '' and old_row['on_2b'] != next_event['on_2b']):
							# runner advances to 3B
							if (old_row['on_2b'] == next_event['on_3b']):
								net_bases += 1
							elif (net_score > 0):
								net_bases += 2
								net_score -= 1
							elif (net_outs > 0):
								net_bases -= 2
								net_outs -= 1

						# check 1B
						if (old_row['on_1b'] != '' and old_row['on_1b'] != next_event['on_1b']):
							# runner advances to 3B
							if (old_row['on_1b'] == next_event['on_2b']):
								net_bases += 1
							elif (old_row['on_1b'] == next_event['on_3b']):
								net_bases += 2
							elif (net_score > 0):
								net_bases += 3
								net_score -= 1
							elif (net_outs > 0):
								net_bases -= 1
								net_outs -= 1

						# check batter
						if (old_row['batter'] != next_event['batter']):
							# runner advances to 3B
							if (old_row['batter'] == next_event['on_1b']):
								net_bases += 1
							elif (old_row['batter'] == next_event['on_2b']):
								net_bases += 2
							elif (old_row['batter'] == next_event['on_3b']):
								net_bases += 3
							elif (net_score > 0):
								net_bases += 4
								net_score -= 1
							elif (net_outs > 0):
								net_outs -= 1

						# write down net totals
						new_row['total_bases'] = net_bases

					writer.writerow(new_row)
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


# getInfo(all_data_file_name)
csvPreProcessing(all_data_file_name, processed_data_file_name)

# EB looks for an 'application' callable by default.
# application = Flask(__name__)

# run the app.
# if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    # application.run()