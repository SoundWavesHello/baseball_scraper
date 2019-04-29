import csv
from pybaseball import playerid_reverse_lookup

input_files = ['2018_final.csv']
years = ["2015", "2016", "2017", "2018"]
individuals = ["betts", "cano", "chapman", "hosmer", "kiermaier", "simmons", "trout"]

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

current = individuals_files(years, individuals)
finalize(current)