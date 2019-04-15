import tensorflow as tf
import numpy
import csv

input_files = ["processed_data_2015.csv"]

def make_prediction(given):
	model_path = "checkpoints/model.ckpt-10000"
	detection_graph = tf.Graph()
	with tf.Session(graph=detection_graph) as sess:
		# Load the graph with the trained states
		loader = tf.train.import_meta_graph(model_path+'.meta')
		loader.restore(sess, model_path)
		_scores = sess.run([], feed_dict = {output_tensor: given})
		return _scores

def main():
	players = {}
	diamond_dict = {"1": "fielder_1", "2": "fielder_2", "3": "fielder_3", "4": "fielder_4", "5": "fielder_5", "6": "fielder_6", "7": "fielder_7", "8": "fielder_8", "9": "fielder_9"}
	
	for input_file in input_files:
		with open(input_file) as file_r:
			reader = csv.DictReader(file_r)
			for row in reader:
				# sets the on_base variables to binary
				on_1 = 0 if row['on_1b'] == "" else 1
				on_2 = 0 if row['on_2b'] == "" else 1
				on_3 = 0 if row['on_3b'] == "" else 1

				if not(row['hc_x'] == "" or row['hc_y'] == "" or row['launch_angle'] == "" or row['launch_speed'] == "" or row['estimated_ba_using_speedangle'] == "" or row['outs_when_up'] == "" or row['total_bases'] == ""): # add or row['home_team'] == ""
					new_input = [float(row['hc_x']), float(row['hc_y']), float(row['launch_angle']), float(row['launch_speed']), float(row['estimated_ba_using_speedangle']), int(float(row['outs_when_up'])), on_1, on_2, on_3,  int(float(row['total_bases']))] # row['home_team'],  # team_dict[row['home_team']],  
				
					print(make_prediction(new_input))

				# # determine what we'll be modifying in the dictionary
				# player = row[diamond_dict[row['hit_location']]]
				# if player not in players.keys():
				# 	players[player] = make_prediction(new_input) - 6 - int(float(row['total_bases']))
				# else:
				# 	players[player] += make_prediction(new_input) - 6 - int(float(row['total_bases']))

		file_r.close()

	for item in players:
		print(item)

main()