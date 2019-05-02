import csv
from pybaseball import playerid_reverse_lookup
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


input_files = ['2018_final.csv']
years = ["2015", "2016", "2017", "2018"]
individuals = ["betts", "cano", "chapman", "hosmer", "kiermaier", "simmons", "trout"]


array = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,849,25,32,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,632,42,10,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1224,60,16,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,62876,1821,907,13,0,0,0,0,2,0,0],[0,0,0,0,0,2,7751,8365,1092,24,0,0,0,0,1,0,1],[0,0,0,0,0,0,4858,3427,3020,34,0,0,0,0,2,0,0],[0,0,0,0,0,0,1389,1377,662,42,0,0,0,0,1,0,0],[0,0,0,0,0,0,484,196,589,10,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

def finalize(input_files):
	header_dict = {"last_name": 1, "first_name": 1, "position": 1, "total_bases": 1, "opportunities": 1, "percentage": 1, "player_id": 1}
	for file in input_files:
		with open(file) as file_r:
			output_file = file[:-4] + "_final.csv"
			with open(output_file, 'a') as file_w:

				reader = csv.DictReader(file_r)
				writer = csv.DictWriter(file_w, header_dict.keys())
				writer.writeheader()

				data = list(reader)

				for old_row in data:
					new_row = {}
					# print(old_row.keys())
					if old_row['player_id'] == "":
						continue
					lookup = int(float(old_row['player_id']))
					data = playerid_reverse_lookup([lookup], key_type="mlbam")
					new_row['player_id'] = lookup
					new_row['position'] = old_row['position']
					new_row['last_name'] = data['name_last'][0]
					new_row['first_name'] = data['name_first'][0]
					new_row['total_bases'] = old_row['total_bases']
					new_row['opportunities'] = old_row['opportunities']

					percentage = int(new_row['total_bases']) / int(new_row['opportunities'])
					new_row['percentage'] = round(percentage, 3)
					# print(new_row)
					writer.writerow(new_row)
			file_w.close()
		file_r.close()


def min_opportunties(num_min, input_files):
	header_dict = {"last_name": 1, "first_name": 1, "position": 1, "total_bases": 1, "opportunities": 1, "percentage": 1, "player_id": 1}
	for file in input_files:
		with open(file) as file_r:
			output_file = file[:-4] + "_" + str(num_min) + ".csv"
			with open(output_file, 'a') as file_w:

				reader = csv.DictReader(file_r)
				writer = csv.DictWriter(file_w, header_dict.keys())
				writer.writeheader()

				data = list(reader)

				for old_row in data:
					if int(old_row['opportunities']) < num_min:
						continue
					new_row = {}
					new_row['last_name'] = old_row['last_name']
					new_row['first_name'] = old_row['first_name']
					new_row['position'] = old_row['position']
					new_row['total_bases'] = old_row['total_bases']
					new_row['opportunities'] = old_row['opportunities']
					new_row['percentage'] = old_row['percentage']
					new_row['player_id'] = old_row['player_id']

					writer.writerow(new_row)
			file_w.close()
		file_r.close()


def individuals_files(years, people):
	file_names = []
	for year in years:
		for person in people:
			name = year + "_" + person + ".csv"
			file_names.append(name)
	return file_names

# current = individuals_files(years, individuals)
# finalize(current)


def calc_replacement(input_files):
	base_tally = {}
	opportunity_tally = {}
	replacements ={}
	for file in input_files:
		with open(file) as file_r:
			reader = csv.DictReader(file_r)

			data = list(reader)
			for row in data:
				if row['position'] not in base_tally.keys():
					base_tally[row['position']] = int(row['total_bases'])
					opportunity_tally[row['position']] = int(row['opportunities'])
				else:
					base_tally[row['position']] += int(row['total_bases'])
					opportunity_tally[row['position']] += int(row['opportunities'])
				# print(base_tally)

	print(base_tally)
	print(opportunity_tally)
	for position in base_tally.keys():
		base_per_opp = base_tally[position] / opportunity_tally[position]
		replacements[position] = round(base_per_opp, 3)

	print(replacements)


def create_plot(confusion_matrix):
	df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(-6,11)],columns = [i for i in range(-6,11)])

	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)

# min_opportunties(100, input_files)
# calc_replacement(input_files)

create_plot(array)
